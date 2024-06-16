# import matplotlib.pyplot as plt

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generative.networks.schedulers import PNDMScheduler
from torch.cuda.amp import autocast
from torch.nn.functional import pad

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.utils.simplex_noise import generate_simplex_noise
from src.utils.visualize_rf import visualize_original_vs_reconstructed

from .base import BaseTrainer


def shuffle(x):
    return np.transpose(x.cpu().numpy(), (1, 2, 0))


class Reconstruct(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.out_dir = self.run_dir / "ood"
        self.out_dir.mkdir(exist_ok=True)
        self.in_channels = args.in_channels
        self.model_type = args.model_type

        # set up loaders
        self.val_loader = get_training_data_loader(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            training_h5file=args.validation_h5file,
            validation_h5file=args.validation_h5file,
            only_val=True,
            num_workers=args.num_workers,
            num_val_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            drop_last=bool(args.drop_last),
            first_n=int(args.first_n_val) if args.first_n_val else args.first_n_val,
            # spatial_dimension=args.spatial_dimension,
        )

        self.in_loader = get_training_data_loader(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            training_h5file=args.in_ids,
            validation_h5file=args.in_ids,
            only_val=True,
            num_workers=args.num_workers,
            num_val_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            drop_last=bool(args.drop_last),
            first_n=int(args.first_n) if args.first_n else args.first_n,
            # spatial_dimension=args.spatial_dimension,
        )

    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )
                if self.snr_shift != 1:
                    snr = pndm_scheduler.alphas_cumprod / (1 - pndm_scheduler.alphas_cumprod)
                    target_snr = snr * self.snr_shift
                    new_alphas_cumprod = 1 / (torch.pow(target_snr, -1) + 1)
                    new_alphas = torch.zeros_like(new_alphas_cumprod)
                    new_alphas[0] = new_alphas_cumprod[0]
                    for i in range(1, len(new_alphas)):
                        new_alphas[i] = new_alphas_cumprod[i] / new_alphas_cumprod[i - 1]
                    new_betas = 1 - new_alphas
                    pndm_scheduler.betas = new_betas
                    pndm_scheduler.alphas = new_alphas
                    pndm_scheduler.alphas_cumprod = new_alphas_cumprod
                pndm_scheduler.set_timesteps(100)
                pndm_timesteps = pndm_scheduler.timesteps
                pndm_start_points = reversed(pndm_timesteps)[1::inference_skip_factor]

                t1 = time.time()
                signals_original = batch["signal"].to(self.device)
                signals = self.vqvae_model.encode_stage_2_inputs(signals_original)
                if self.do_latent_pad:
                    signals = F.pad(input=signals, pad=self.latent_pad, mode="constant", value=0)

                if self.model_type == "autoencoder":

                    reconstructions = self.model(signals)

                    non_batch_dims = tuple(range(signals_original.dim()))[1:]
                    mse_metric = torch.square(signals_original - reconstructions).mean(axis=non_batch_dims)
                    for b in range(signals.shape[0]):
                        stem = b

                        results.append(
                            {
                                "filename": stem,
                                "type":     dataset_name,
                                "t":        0,
                                "mse":      mse_metric[b].item(),
                            }
                        )
                    visualize_original_vs_reconstructed(
                        signals_original.cpu(),
                        reconstructions.cpu(),
                        N=3,
                    )


                else:
                    # loop over different values to reconstruct from
                    for t_start in pndm_start_points:
                        with autocast(enabled=True):
                            start_timesteps = torch.Tensor([t_start] * signals.shape[0]).long()

                            # noise signals
                            if self.simplex_noise:
                                noise = generate_simplex_noise(
                                    self.simplex,
                                    x=signals,
                                    t=start_timesteps,
                                    in_channels=signals.shape[1],
                                )
                            else:
                                noise = torch.randn_like(signals).to(self.device)

                            reconstructions = pndm_scheduler.add_noise(
                                original_samples=signals * self.b_scale,
                                noise=noise,
                                timesteps=start_timesteps,
                            )
                            # perform reconstruction
                            for step in pndm_timesteps[pndm_timesteps <= t_start]:
                                timesteps = torch.Tensor([step] * signals.shape[0]).long()
                                model_output = self.model(
                                    reconstructions, timesteps=timesteps.to(self.device)
                                )
                                # 2. compute previous signal: x_t -> x_t-1
                                reconstructions, _ = pndm_scheduler.step(
                                    model_output, step, reconstructions
                                )
                        # try clamping the reconstructions
                        if self.do_latent_pad:
                            reconstructions = F.pad(
                                input=reconstructions,
                                pad=self.inverse_latent_pad,
                                mode="constant",
                                value=0,
                            )
                        reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                        reconstructions = reconstructions / self.b_scale
                        # reconstructions.clamp_(0, 1) # FFUUUUKKKKK

                        non_batch_dims = tuple(range(signals_original.dim()))[1:]
                        mse_metric = torch.square(signals_original - reconstructions).mean(
                            axis=non_batch_dims
                        )
                        for b in range(signals.shape[0]):
                            stem = b

                            results.append(
                                {
                                    "filename": stem,
                                    "type":     dataset_name,
                                    "t":        t_start.item(),
                                    "mse":      mse_metric[b].item(),
                                }
                            )
                        # plot
                        # if not dist.is_initialized():
                            # visualize_original_vs_reconstructed(
                            #     signals_original.cpu(),
                            #     reconstructions.cpu(),
                            #     N=3,
                            # )
                t2 = time.time()
                if dist.is_initialized():
                    print(f"{dist.get_rank()}: Took {t2 - t1}s for a batch size of {signals.shape[0]}")
                else:
                    print(f"Took {t2 - t1}s for a batch size of {signals.shape[0]}")
        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_val):
            results_list = self.get_scores(self.val_loader, "val", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            results_list = self.get_scores(self.in_loader, "in", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_in.csv")

        if bool(args.run_out):
            for out in args.out_ids.split(","):
                print(out)

                out_loader = get_training_data_loader(
                    dataset_name=args.dataset_name,
                    batch_size=args.batch_size,
                    training_h5file=out,
                    validation_h5file=out,
                    only_val=True,
                    num_workers=args.num_workers,
                    num_val_workers=args.num_workers,
                    cache_data=bool(args.cache_data),
                    drop_last=bool(args.drop_last),
                    first_n=int(args.first_n) if args.first_n else args.first_n,
                    # spatial_dimension=args.spatial_dimension,
                )
                dataset_name = Path(out).stem.split("_")[0]
                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")
