import os
import h5py
import numpy as np
import argparse


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


# Función para crear la nueva estructura de directorios
def create_new_directory_structure(base_path, new_base_path):
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            new_dir = root.replace(base_path, new_base_path, 1)
            new_dir = os.path.join(new_dir, dir)
            os.makedirs(new_dir, exist_ok=True)


# Función para procesar un archivo h5
def process_h5_file(file_path, new_file_path, trim_length, num_samples):
    with h5py.File(file_path, 'r') as f:
        dataset = f['dataset'][:]
        sig_type = f['sig_type'][()]

        num_original_samples, original_length = dataset.shape
        new_samples = []

        # Crear nuevas muestras dividiendo las existentes
        for i in range(num_original_samples):
            start_idx = 0
            while start_idx + trim_length <= original_length and len(new_samples) < num_samples:
                new_samples.append(dataset[i, start_idx:start_idx + trim_length])
                start_idx += trim_length

        new_samples = np.array(new_samples)

        # Añadir una dimensión para la parte real e imaginaria
        dataset_expanded = np.stack((new_samples.real, new_samples.imag), axis=-1)

    with h5py.File(new_file_path, 'w') as f:
        f.create_dataset('dataset', data=dataset_expanded, dtype=np.float64)
        f.create_dataset('sig_type', data=sig_type)


# Función principal para recorrer la estructura de directorios y procesar los archivos
def preprocess(base_path, new_base_path, trim_length, num_samples):
    create_new_directory_structure(base_path, new_base_path)

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                new_root = root.replace(base_path, new_base_path, 1)
                new_file_path = os.path.join(new_root, file)
                print(f"Processing {file_path}...")
                process_h5_file(file_path, new_file_path, trim_length, num_samples)
                print(f"Finished processing {file_path}")


# Funcion main para mostrar directorio inicial, aplicar la funcion preprocess y mostrar directorio final
# Usa argparse para recibir los parametros de entrada

def main():
    parser = argparse.ArgumentParser(description='Preprocess signal data')
    parser.add_argument('--data_root', type=str, help='Path to the directory containing the signal data')
    parser.add_argument('--new_data_root', type=str, help='Path to the directory where the preprocessed data will be saved')
    parser.add_argument('--trim_length', type=int, default=3000, help='Length to trim the signals')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to get from each signal')
    args = parser.parse_args()

    print('Initial directory structure:')
    display_directory_structure_with_h5_info(args.data_root)
    print()
    print(f"Trimming signals to length {args.trim_length} and getting {args.num_samples} samples from each signal...")
    preprocess(args.data_root, args.new_data_root, args.trim_length, args.num_samples)
    print()
    print('Final directory structure:')
    display_directory_structure_with_h5_info(args.new_data_root)


if __name__ == '__main__':
    main()
