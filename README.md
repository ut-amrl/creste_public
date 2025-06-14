# A Framework for Scalable Mapless Navigation

This project implements **CREStE**, a scalable framework for open-world local planning using visual foundation models and counterfactual guidance. This work is published in the **Robotics Science and Systems (RSS) 2025** conference proceedings.

## CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance

[![Website](docs/badges/badge-website.svg)](https://amrl.cs.utexas.edu/creste)
[![Paper](docs/badges/badge-pdf.svg)](https://arxiv.org/abs/2503.03921)

Team:
[Arthur Zhang](https://www.arthurkzhang.com/),
[Harshit Sikchi](https://hari-sikchi.github.io),
[Amy Zhang](https://amyzhang.github.io)
[Joydeep Biswas](https://www.joydeepb.com),

We present CREStE, a scalable framework for mapless navigation that leverages internet scale priors from visual foundation models and counterfactual guidance for open-world local path planning. Notably, CREStE does not require exhaustively pre-enumerated lists of semantic classes and generalizes to novel urban environments with just 3 hours of expert demonstrations, and can be improved offline by simpling providing additional counterfactual annotations. Our approach runs in real-time on a single laptop GPU at 20Hz and acheives state-of-the-art performance, generalizing robustly to novel environments with just 3 hours of expert demonstrations.

# 📢 News

- 2025-06-14: Inital code and pretrained model release.
- 2025-05-10: Spotlight Oral Presentation at ICRA Safe VLM Workshop 2025.
- 2025-04-18: Spotlight Oral Presentation at Texas Regional Robotics Symposium (TEROS) 2025.
- 2025-04-10: Paper accepted to Robotics Science and Systems (RSS) 2025.
- 2025-03-04: Best Student Paper Award at UT AI x Robotics Symposium 2025.
- 2025-03-01: Initial website and paper release.

# 🚀 Usage

Despite being trained on a single robot embodiment on just 3 hours of data, CREStE generalizes remarkably well to novel embodiments and sensor configurations. To download the pretrained model weights for the monocular RGB + LiDAR modality, run the commands below:

```bash
# Monocular RGB + LiDAR
bash ./scripts/release/download_weights.sh creste_rgbd
```

To run the CREStE model in realtime, we recommend using our sister repository: [creste_realtime](https://github.com/ut-amrl/creste_realtime). This repository provides an efficient C++ implementation of the inference pipeline in ROS1, free from any complex python dependencies. 

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

After preprocessing the dataset, you can start training the CREStE model. We provide full details on how to train the model in the [Training](./docs/TRAINING.md) section. At a high level, we train CREStE in three steps:
1. **RGB-D Backbone**: Train the RGB-D backbone using the Dinov2 distillation method.
2. **BEV Backbone**: Train the BEV backbone using the Dinov2 distillation method with SAM2 instance labels and elevation maps.
3. **Reward Function**: Train the reward function using MaxEnt IRL with the pretrained BEV backbone, and then refine it with counterfactual annotations using Counterfactual IRL.

After training CREStE, you can compile it for use with our realtime inference pipeline in C++. More instructions on this can be found in the last section of the [Training](./docs/TRAINING.md) section.

