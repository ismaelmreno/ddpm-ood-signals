<h1 align="center">Denoising diffusion models for out-of-distribution detection</h1>
<p align="center">
Perform reconstruction-based out-of-distribution detection with DDPMs.
</p>

## Intro

This codebase contains the code to perform unsupervised out-of-distribution detection with diffusion models.
It supports the use of DDPMs as well as Latent Diffusion Models (LDM) for dealing with higher dimensional 2D or 3D data.
It is based on work published in [1] and [2].

[1] [Denoising diffusion models for out-of-distribution detection, CVPR VAND Workshop 2023](https://arxiv.org/abs/2211.07740)

[2] [Unsupervised 3D out-of-distribution detection with latent diffusion models, MICCAI 2023](https://arxiv.org/abs/2307.03777)

## Setup

### Install

Create a fresh virtualenv (this codebase was developed and tested with Python 3.8) and then install the required
packages:

```pip install -r requirements.txt```

[//]: # (You can also build the docker image)

[//]: # ()

[//]: # (```bash)

[//]: # (cd docker/)

[//]: # (bash create_docker_image.sh)

[//]: # (```)

### Setup paths

Select where you want your data and model outputs stored.

```bash
data_root=/home/ismael/TFM/rfchallenge/
output_root=/home/ismael/TFM/rfmodelsoutput/
```

## Run with DDPM

We'l use signal datasets ...

### Download and process datasets

```bash
wget -O  ${data_root}/dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
unzip  ${data_root}/dataset.zip -d ${data_root}
rm ${data_root}/dataset.zip
```

The following code will reshape and trim the signal datasets to the desired length and number of samples, as well as
unpacking real and imaginary parts of the signal into two separate channels.

```bash
python src/data/signal_trim_and_unpack.py --data_root=${data_root}/dataset --trim_length=3000 --new_data_root=${data_root}/dataset_processed
```

### Train models

Examples here use FashionMNIST as the in-distribution dataset. Commands for other datasets are given
in [README_additional.md](README_additional.md).

```bash
python train_ddpm.py \
--output_dir=${output_root} \
--model_name=unet_medium_EMISignal1 \
--model_type=medium \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_val.h5 \
--n_epochs=2 \
--beta_start=0.0015 \
--beta_end=0.0195 \
--batch_size=5 \
--quick_test=0

```

--beta_schedule=scaled_linear \

You can track experiments in tensorboard

```bash
tensorboard --logdir=${output_root}
```

The code is DistributedDataParallel (DDP) compatible. To train on e.g. 2 GPUs:

```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
train_ddpm.py \
--output_dir=${output_root} \
--model_name=test3 \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_val.h5 \
--n_epochs=30 \
--beta_start=0.0015 \
--beta_end=0.0195 \
--batch_size=5 \
--quick_test=0
```

#### Train a base model - Autoencoder

```bash
python train_autoencoder.py \
--output_dir=${output_root} \
--model_name=autoencoder_EMISignal1 \
--model_type=autoencoder \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_val.h5 \
--n_epochs=400 \
--batch_size=40 \
--quick_test=0
```

### Reconstruct data

list of files
CommSignal2
CommSignal3
CommSignal5G1
EMISignal1


```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=wavenet_CommSignal2 \
--model_type=wavenet \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_val.h5 \
--in_ids=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_val.h5 \
--out_ids=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_val.h5,${data_root}/dataset_processed/interferenceset_frame/CommSignal3_raw_data_val.h5,${data_root}/dataset_processed/interferenceset_frame/CommSignal5G1_raw_data_val.h5 \
--beta_start=0.0015 \
--beta_end=0.0195 \
--num_inference_steps=100 \
--inference_skip_factor=4 \
--run_val=0 \
--run_in=1 \
--batch_size=50 \
--run_out=1 \
--first_n=50 \
--first_n_val=50


```




The arg `inference_skip_factor` controls the amount of t starting points that are skipped during reconstruction.
This table shows the relationship between values of `inference_skip_factor` and the number of reconstructions, as needed
to reproduce results in Supplementary Table 4 (for max_t=1000).

| **inference_skip_factor:** | 1   | 2  | 3  | 4  | 5  | 8  | 16 | 32 | 64 |
|----------------------------|-----|----|----|----|----|----|----|----|----|
| **num_reconstructions:**   | 100 | 50 | 34 | 25 | 20 | 13 | 7  | 4  | 2  |

N.B. For a quicker run, you can choose to only reconstruct a subset of the validation set with e.g. `--first_n_val=1000`
or a subset of the in/out datasets with `--first_n=1000`

### Classify samples as OOD

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=unet_medium_CommSignal2
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=unet_medium_CommSignal3
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=unet_medium_CommSignal5G1
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=unet_medium_EMISignal1
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=wavenet_CommSignal2
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=wavenet_CommSignal3
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=wavenet_CommSignal5G1
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=wavenet_EMISignal1
```






## Run with LDM

### Train VQVAE



```bash
python train_vqvae.py  \
--output_dir=${output_root} \
--model_name=vqvae_CommSignal2 \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_val.h5 \
--n_epochs=3 \
--batch_size=8  \
--eval_freq=100 \
--cache_data=0  \
--spatial_dimension=1
```

```bash
python train_vqvae.py  \
--output_dir=${output_root} \
--model_name=vqvae_CommSignal3 \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal3_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal3_raw_data_val.h5 \
--n_epochs=1 \
--batch_size=8  \
--eval_freq=100 \
--cache_data=0  \
--spatial_dimension=1
```

```bash
python train_vqvae.py  \
--output_dir=${output_root} \
--model_name=vqvae_CommSignal5G1 \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal5G1_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal5G1_raw_data_val.h5 \
--n_epochs=1 \
--batch_size=8  \
--eval_freq=100 \
--cache_data=0  \
--spatial_dimension=1
```
```bash
python train_vqvae.py  \
--output_dir=${output_root} \
--model_name=vqvae_EMISignal1 \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_val.h5 \
--n_epochs=1 \
--batch_size=8  \
--eval_freq=100 \
--cache_data=0  \
--spatial_dimension=1
```

The code is DistributedDataParallel (DDP) compatible. To train on e.g. 2 GPUs run with
`torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train_vqvae.py`

### Train LDM

```bash
python train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=unetvqvae_CommSignal2 \
  --vqvae_checkpoint=${output_root}/vqvae_CommSignal2/checkpoint.pth \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal2_raw_data_val.h5 \
  --n_epochs=20 \
  --batch_size=6 \
  --eval_freq=25 \
  --checkpoint_every=1 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=medium \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=1
```

```bash
python train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=unetvqvae_CommSignal3 \
  --vqvae_checkpoint=${output_root}/vqvae_CommSignal3/checkpoint.pth \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal3_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal3_raw_data_val.h5 \
  --n_epochs=20 \
  --batch_size=6 \
  --eval_freq=25 \
  --checkpoint_every=1 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=medium \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=1
```

```bash
python train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=unetvqvae_CommSignal5G1 \
  --vqvae_checkpoint=${output_root}/vqvae_CommSignal5G1/checkpoint.pth \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal5G1_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/CommSignal5G1_raw_data_val.h5 \
  --n_epochs=20 \
  --batch_size=6 \
  --eval_freq=25 \
  --checkpoint_every=1 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=medium \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=1
```

```bash
python train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=unetvqvae_EMISignal1 \
  --vqvae_checkpoint=${output_root}/vqvae_EMISignal1/checkpoint.pth \
--training_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_train.h5 \
--validation_h5file=${data_root}/dataset_processed/interferenceset_frame/EMISignal1_raw_data_val.h5 \
  --n_epochs=20 \
  --batch_size=6 \
  --eval_freq=25 \
  --checkpoint_every=1 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=medium \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=1
```



### Reconstruct data

```bash
python reconstruct.py \
  --output_dir=${output_root} \
  --model_name=ddpm_decathlon \
  --vqvae_checkpoint=${output_root}/decathlon-vqvae-4layer/checkpoint.pth \
  --validation_h5file=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
  --in_ids=${data_root}/data_splits/Task01_BrainTumour_test.csv \
  --out_ids=${data_root}/data_splits/Task02_Heart_test.csv,${data_root}/data_splits/Task03_Liver_test.csv,${data_root}/data_splits/Task04_Hippocampus_test.csv,${data_root}/data_splits/Task05_Prostate_test.csv,${data_root}/data_splits/Task06_Lung_test.csv,${data_root}/data_splits/Task07_Pancreas_test.csv,${data_root}/data_splits/Task08_HepaticVessel_test.csv,${data_root}/data_splits/Task09_Spleen_test.csv\
  --is_grayscale=1 \
  --batch_size=32 \
  --cache_data=0 \
  --prediction_type=epsilon \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --image_roi=[160,160,128] \
  --image_size=128 \
  --num_inference_steps=100 \
  --inference_skip_factor=2 \
  --run_val=1 \
  --run_in=1 \
  --run_out=1 
````

### Classify samples as OOD

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=ddpm_decathlon
```

## Acknowledgements

Built with [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels)
and [MONAI](https://github.com/Project-MONAI/MONAI).

## Citations

If you use this codebase, please cite

```bib
@InProceedings{Graham_2023_CVPR,
    author    = {Graham, Mark S. and Pinaya, Walter H.L. and Tudosiu, Petru-Daniel and Nachev, Parashkev and Ourselin, Sebastien and Cardoso, Jorge},
    title     = {Denoising Diffusion Models for Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2947-2956}
}
@inproceedings{graham2023unsupervised,
  title={Unsupervised 3D out-of-distribution detection with latent diffusion models},
  author={Graham, Mark S and Pinaya, Walter Hugo Lopez and Wright, Paul and Tudosiu, Petru-Daniel and Mah, Yee H and Teo, James T and J{\"a}ger, H Rolf and Werring, David and Nachev, Parashkev and Ourselin, Sebastien and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={446--456},
  year={2023},
  organization={Springer}
}
}
```
