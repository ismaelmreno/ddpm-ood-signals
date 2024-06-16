import argparse
import ast

from src.trainers import VQVAETrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_h5file", help="Location of file with training ids.")
    parser.add_argument("--validation_h5file", help="Location of file with validation ids.")
    parser.add_argument(
        "--spatial_dimension", default=1, type=int, help="Dimension of images: 2d or 3d."
    )

    # training param
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--dataset_name", default="dataset", help="Name of dataset in h5 file.")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Number of epochs to between evaluations.",
    )
    parser.add_argument(
        "--augmentation",
        type=int,
        default=1,
        help="Use of augmentation, 1 (True) or 0 (False).",
    )
    parser.add_argument(
        "--adversarial_weight",
        type=float,
        default=0.01,
        help="Weight for adversarial component.",
    )
    parser.add_argument(
        "--adversarial_warmup",
        type=int,
        default=0,
        help="Warmup the learning rate of the adversarial component.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=1,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save a checkpoint every checkpoint_every epochs.",
    )
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    args = parser.parse_args()
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_ddpm.py --args
if __name__ == "__main__":
    args = parse_args()
    trainer = VQVAETrainer(args)
    trainer.train(args)
