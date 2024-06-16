import os
import h5py
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def list_h5_contents_with_details(file_path):
    contents = []
    with h5py.File(file_path, 'r') as h5_file:
        def collect_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                contents.append(f"{name}: {obj.shape}, dtype={obj.dtype}")

        h5_file.visititems(collect_attrs)
    return contents


def display_directory_structure_with_h5_info(base_dir, indent=0):
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            print('│   ' * indent + '├── ' + item)
            display_directory_structure_with_h5_info(item_path, indent + 1)
        elif item.endswith('.h5'):
            print('│   ' * indent + '├── ' + item)
            h5_contents = list_h5_contents_with_details(item_path)
            for content in h5_contents:
                print('│   ' * (indent + 1) + '├── ' + content)


def create_new_directory_structure(base_path, new_base_path):
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            new_dir = root.replace(base_path, new_base_path, 1)
            new_dir = os.path.join(new_dir, dir)
            os.makedirs(new_dir, exist_ok=True)


def process_h5_file(file_path, train_file_path, val_file_path, trim_length, num_samples, val_split):
    with h5py.File(file_path, 'r') as f:
        dataset = f['dataset'][:]
        sig_type = f['sig_type'][()]

        num_original_samples, original_length = dataset.shape
        new_samples = []

        for i in range(num_original_samples):
            start_idx = 0
            while start_idx + trim_length <= original_length:
                new_samples.append(dataset[i, start_idx:start_idx + trim_length])
                start_idx += trim_length
                if num_samples is not None and len(new_samples) >= num_samples:
                    break
            if num_samples is not None and len(new_samples) >= num_samples:
                break

        new_samples = np.array(new_samples)
        dataset_expanded = np.stack((new_samples.real, new_samples.imag), axis=1)

        train_data, val_data = train_test_split(dataset_expanded, test_size=val_split)

    with h5py.File(train_file_path, 'w') as f:
        f.create_dataset('dataset', data=train_data, dtype=np.float32)
        f.create_dataset('sig_type', data=sig_type)

    with h5py.File(val_file_path, 'w') as f:
        f.create_dataset('dataset', data=val_data, dtype=np.float32)
        f.create_dataset('sig_type', data=sig_type)


def preprocess(base_path, new_base_path, trim_length, num_samples, val_split):
    create_new_directory_structure(base_path, new_base_path)

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                new_root = root.replace(base_path, new_base_path, 1)
                base_file_name = os.path.splitext(file)[0]
                train_file_path = os.path.join(new_root, base_file_name + '_train.h5')
                val_file_path = os.path.join(new_root, base_file_name + '_val.h5')
                print(f"Processing {file_path}...")
                process_h5_file(file_path, train_file_path, val_file_path, trim_length, num_samples, val_split)
                print(f"Finished processing {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess signal data')
    parser.add_argument('--data_root', type=str, help='Path to the directory containing the signal data')
    parser.add_argument('--new_data_root', type=str, help='Path to the directory where the preprocessed data will be saved')
    parser.add_argument('--trim_length', type=int, default=3000, help='Length to trim the signals')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to get from each signal (None for maximum possible samples)')
    parser.add_argument('--val_split', type=float, default=0.3, help='Proportion of the dataset to include in the validation split')
    args = parser.parse_args()

    print('Initial directory structure:')
    display_directory_structure_with_h5_info(args.data_root)
    print()
    print(
        f"Trimming signals to length {args.trim_length}, getting {args.num_samples if args.num_samples is not None else 'maximum possible'} samples from each signal, and splitting with validation split of {args.val_split}...")
    preprocess(args.data_root, args.new_data_root, args.trim_length, args.num_samples, args.val_split)
    print()
    print('Final directory structure:')
    display_directory_structure_with_h5_info(args.new_data_root)


if __name__ == '__main__':
    main()
