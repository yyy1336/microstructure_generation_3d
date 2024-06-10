# Guided Diffusion for Fast Inverse Design of Density-based Mechanical Metamaterials

## Installation
Following is the suggested way to install the dependencies of our code:
```
conda create -n mg3d python=3.8
conda activate mg3d

conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=10.2 -c pytorch -c nvidia

pip install tqdm fire einops pyrender pyrr trimesh timm scikit-image==0.18.2 scikit-learn==0.24.2 pytorch-lightning==1.6.1
```
## Usage
We provide three simple use cases in scripts. You need to change --dataset_folder, --result_folder, --output_path, and --model_path to your own paths, specifying the dataset location, training result output location, generation result output location, and the model used for inference, respectively.

## Dataset
You can download each generation of dataset from [Google Drive](https://drive.google.com/drive/folders/1fNj_v-8YjtYCPoyXn6qZ-HzG0LqAJeV9?usp=drive_link).

## Pre-trained Model
You can download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1sODtlQTbZubZd8Vh4jAqHDHAPimyNeiH?usp=drive_link).

## Rawdata
You can download rawdata for some results in the article from [Google Drive](https://drive.google.com/drive/folders/1zz94C8dDDqNO5EEhqe9d3vd37KWO1Eu9?usp=drive_link).
