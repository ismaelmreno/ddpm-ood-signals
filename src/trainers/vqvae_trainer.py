import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import L1Loss

# from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.networks.vqvae_signal.convolutional_vq_vae import ConvolutionalVQVAE


class VQVAETrainer:
    def __init__(self, args):

        # initialise DDP if run was launched with torchrun
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f

            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.ddp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # set up model
        self.spatial_dimension = args.spatial_dimension
        # def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        config = {
            'input_features_dim':      2,  # Dimensión de las características de entrada
            'output_features_dim':     2,  # Dimensión de las características de salida
            'num_hiddens':             8,  # Número de canales ocultos en el encoder y decoder
            'num_residual_layers':     2,  # Número de capas residuales
            'residual_channels':       64,  # Número de canales en las capas residuales
            'use_kaiming_normal':      True,  # Usar inicialización Kaiming normal
            'input_features_filters':  2,  # Filtros de características de entrada
            'augment_input_features':  False,  # No aumentar las características de entrada
            'output_features_filters': 2,  # Filtros de características de salida
            'augment_output_features': False,  # No aumentar las características de salida
            'num_embeddings':          512,  # Número de embeddings en el vector quantizer
            'embedding_dim':           8,  # Dimensión de los embeddings
            'commitment_cost':         0.25,  # Costo de compromiso para el vector quantizer
            'decay':                   0.99,  # Decaimiento para VectorQuantizerEMA
            'use_jitter':              True,  # Usar jitter en el decoder
            'jitter_probability':      0.12,  # Probabilidad de aplicar jitter
            'record_codebook_stats':   True,  # Registrar estadísticas del código
            'verbose':                 False  # Activar modo detallado para salida de logs
        }

        self.model = ConvolutionalVQVAE(config, self.device)
        self.model.to(self.device)
        print(f"{sum(p.numel() for p in self.model.parameters()):,} model parameters")

        # set up optimizer, loss, checkpoints
        self.run_dir = Path(args.output_dir) / args.model_name
        checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            print(
                f"Resuming training using checkpoint {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0

        # save vqvae parameters
        self.run_dir.mkdir(exist_ok=True)
        with open(self.run_dir / "vqvae_config.json", "w") as f:
            json.dump(config, f, indent=4)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs
        self.train_loader, self.val_loader = get_training_data_loader(
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            training_h5file=args.training_h5file,
            validation_h5file=args.validation_h5file,
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            # spatial_dimension=args.spatial_dimension,
        )

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch":                epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step":          self.global_step,
                "model_state_dict":     self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss":            self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch":                epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step":          self.global_step,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss":            self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)

    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if args.checkpoint_every != 0 and (epoch + 1) % args.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch + 1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch + 1}",
                )

            if (epoch + 1) % args.eval_freq == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        if self.ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=110,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        loss = 0

        epoch_loss = 0
        epoch_step = 0
        self.model.train()
        for step, batch in progress_bar:
            signals = batch["signal"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            reconstruction, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized = self.model(signals)

            recons_loss = L1Loss()(reconstruction, signals)
            loss = recons_loss + vq_loss
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_step += signals.shape[0]
            self.global_step += signals.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / epoch_step,
                }
            )

            self.logger_train.add_scalar(
                tag="loss", scalar_value=loss.item(), global_step=self.global_step
            )

            if self.quick_test:
                break
        epoch_loss = epoch_loss / epoch_step
        return epoch_loss

    def val_epoch(self, epoch):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                ncols=110,
                position=0,
                leave=True,
                desc="Validation",
            )

            global_val_step = self.global_step
            for step, batch in progress_bar:
                signals = batch["signal"].to(self.device)
                reconstruction, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized = self.model(signals)

                loss = L1Loss()(reconstruction, signals) + vq_loss

                self.logger_val.add_scalar(
                    tag="loss", scalar_value=loss.item(), global_step=global_val_step)

                global_val_step += signals.shape[0]

                # # plot some recons
                # if step == 0:
                #     fig = plt.figure()
                #     for i in range(2):
                #         plt.subplot(2, 2, i * 2 + 1)
                #         if self.spatial_dimension == 2:
                #             sl = np.s_[i, 0, :, :]
                #         else:
                #             sl = np.s_[i, 0, :, :, images.shape[4] // 2]
                #         plt.imshow(images[sl].cpu(), cmap="gray")
                #         if i == 0:
                #             plt.title("Image")
                #         plt.subplot(2, 2, i * 2 + 2)
                #         plt.imshow(reconstruction[sl].cpu(), cmap="gray")
                #         if i == 0:
                #             plt.title("Recon")
                #     plt.show()
                #     self.logger_val.add_figure(
                #         tag="recons", figure=fig, global_step=self.global_step
                #     )
