import pandas as pd
import h5py
import torch.distributed as dist
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset
from torch.utils.data import DataLoader
import numpy as np


def get_data_dicts(h5_file: str, dataset_name: str, shuffle: bool = False, first_n=False):
    """Get data dicts for data loaders."""
    with h5py.File(h5_file, 'r') as f:
        data = f[dataset_name][()]

    if shuffle:
        np.random.seed(1)
        np.random.shuffle(data)

    if first_n:
        data = data[:first_n]

    data_dicts = [{"signal": signal} for signal in data]

    print(f"Found {len(data_dicts)} signals.")

    if dist.is_initialized():
        print(dist.get_rank())
        print(dist.get_world_size())
        return partition_dataset(
            data=data_dicts,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        return data_dicts


def get_training_data_loader(
        batch_size: int,
        training_h5file: str,
        validation_h5file: str,
        dataset_name: str,
        only_val: bool = False,
        drop_last: bool = False,
        num_workers: int = 8,
        num_val_workers: int = 3,
        cache_data=True,
        first_n=None,
):
    # Define a simple identity transformation (could be extended if needed)
    def identity_transform(x):
        return x

    val_dicts = get_data_dicts(validation_h5file, dataset_name, shuffle=False, first_n=first_n)
    if first_n:
        val_dicts = val_dicts[:first_n]

    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=identity_transform,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=identity_transform,
        )

    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(training_h5file, dataset_name, shuffle=True, first_n=first_n)

    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=identity_transform,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=identity_transform,
        )

    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    return train_loader, val_loader

# Example usage:
# train_loader, val_loader = get_training_data_loader(
#     batch_size=32,
#     training_h5='training_data.h5',
#     validation_h5='validation_data.h5',
#     dataset_name='signals',
#     only_val=False,
#     first_n=100,
# )
