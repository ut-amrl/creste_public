# CREStE: Counterfactuals for Reward Enhancement with Structured Embeddings

This project implements CREStE, a scalable framework for open-world local planning using visual foundation models and counterfactual guidance. Using just 3 hours of expert demonstrations, CREStE can generalize to novel urban environments, enabling efficient navigation in complex settings. 

# 📢 News


# 🚀 Usage

Despite being trained on a single robot embodiment on just 3 hours of data, CREStE generalizes remarkably well to novel embodiments and sensor configurations. To download the pretrained model weights for the following modalities, please visit the [Hugging Face model hub](https://huggingface.co/arthurz/crest_e). You may also download the pre-trained torch jit models by running the commands below:

```bash
# Monocular RGB + LiDAR
bash ./scripts/download_weights.sh creste_rgb_lidar
# Stereo RGB
bash ./scripts/download_weights.sh creste_stereo_rgb
# Monocular RGB
bash ./scripts/download_weights.sh creste_mono_rgb
```

To run the CREStE model in realtime, we recommend using our sister repository: [creste_realtime](https://github.com/ut-amrl/creste_realtime). This repository provides an efficient C++ implementation of the inference pipeline in ROS1, free from any complicated python dependencies. 

```bash
git clone https://github.com/ut-amrl/creste_realtime
```

# 🛠️ Setup
For those who wish to train CREStE on their own data, we provide a complete setup guide below for training and evaluating a monocular RGB + LiDAR model below.

The training code was tested on:
- Ubuntu 22.04 with Python 3.10 and PyTorch 2.3.1

## 📦 Repository

```bash
git clone https://github.com/ut-amrl/creste_public
cd creste_public
```

💻 Dependencies

Install the dependencies.

```
conda create -n creste python=3.10
conda activate creste
pip install -e .
```

## 🏃 Preparing Your Dataset

Currently, we only support datasets processed to the [UT CODa dataset format](https://github.com/ut-amrl/coda-devkit/blob/main/docs/DATA_REPORT.md#dataset-organization). We provide a sample dataset for download below to test the training and evaluation code. 

```bash
# Download the sample dataset
bash ./scripts/release/download_dataset.sh
```

The downloaded dataset will be placed in the `data/creste` directory. This dataset should have the following structure:

```
data/creste
├── 2d_rect
    ├── 2d_rect
        ├── cam0
            ├── {seq_id}
                ├── 2d_rect_cam0_{seq_id}_{frame_id}.jpg
                ├── ...
├── 3d_raw
    ├── 3d_raw
        ├── os1
            ├── {seq_id}
                ├── 3d_raw_os1_{seq_id}_{frame_id}.bin
                ├── ...
├── calibrations
    ├── {seq_id}
        ├── calib_cam0_intrinsics.yaml
        ├── calib_os1_to_cam0.yaml
        ├── ...
├── counterfactuals
    ├── {seq_id}
        ├── {frame_id}.pkl
        ├── ...
├── poses
    ├── dense
        ├── {seq_id}.txt
├── splits
    ├── mini
        ├── full.txt
├── timestamps
    ├── {seq_id}.txt
```

Before training, you will need to preprocess the dataset to generate the necessary supervision labels for ground truth depth, Dinov2 feature maps, and BEV semantic and elevation maps. For more information on preprocessing, please refer to the [Data Preparation](./docs/DATA_PREPARATION.md) section.

# 📊 Training



## Datasets

### CODa

To use our models on the UT Campus Object Dataset (CODa), you will need to first create
a data folder and add the CODa files to it. To do this, execute the following commands
from the root directory of this project to symlink CODa.

```bash
mkdir data
ln -s /robodata/arthurz/Research/CompleteNet/data/coda ./data/coda
```

## Creating CODa SSC from scratch

### Build the single frame depth inputs

Depth inputs for semistatic scenes
```bash
python tools/build_dense_depth.py --scans 1 --proc LA --dataset_type semistatic
```

Depth inputs for semantic segmentation scenes
```bash
python tools/build_dense_depth.py --scans 1 --proc LA --dataset_type semanticsegmentation
```

### Build the ground truth depth inputs

We use the IDW infilling method to create the ground truth depth inputs for the semistatic scenes. This method uses semi-global stereo matching to filter accumulated LiDAR point clouds and IDW infilling to further densify the depth input.

Depth labels for semistatic scenes
```bash
python tools/build_dense_depth.py --scans 50 --proc IDW --dataset_type semistatic
```

Depth inputs for semantic segmentation scenes
```bash
python tools/build_dense_depth.py --scans 50 --proc IDW --dataset_type semanticsegmentation
```

Depth inputs for all scenes
```bash
python tools/build_dense_depth.py --scans 50 --proc IDW --dataset_type all
```

### Preprocessing depth images for faster loading

This saves downsampled versions of the input and label depth images to disk for faster loading during training. It results in ~20% faster loading.

Depth labels for semistatic scenes
```bash
python tools/preprocess_dataset.py
```

### Building ground truth semantic segmentation and elevation maps

This builds ground truth semantic scene completion maps for terrain segmentations. It uses annotated lidar
point clouds to create the ground truth labels.

## Using Labels

Semantic Ground Truth
```bash
python tools/build_semantic_map.py \
    --cfg ./configs/dataset/coda.yaml \
    --out_dir ./postprocess/build_map_outputs/codasemantics \
    --save_type semantic \
    --vis True
```

Elevation Ground Truth
```bash
python tools/build_semantic_map.py \
    --cfg ./configs/dataset/coda.yaml \
    --out_dir ./postprocess/build_map_outputs/codaelevation \
    --save_type elevation \
    --vis True
```

## Without Labels
```bash
python tools/build_feature_map.py \
    --cfg ./configs/dataset/distillation/coda_pefree_dinov1.yaml \
    --out_dir ./postprocess/build_map_outputs/elevation \
    --feat_type geometric \
    --tasks elevation \
    --vis True
```

## Computing class frequencies for object detection

Computes sam dynamic general class frequencies
```bash
python tools/compute_weights.py --cfg_file configs/dataset/ssc_sam/coda_sam2elev_joint_ds2.yaml --out_dir data/coda_rlang --task 3d_sam_dynamic
```

## Generating depth images

# 50 LiDAR Accum + SD Filtering
```bash
python tools/build_dense_depth.py --scans 50 --proc PASS --dataset_type single --verbose
```
