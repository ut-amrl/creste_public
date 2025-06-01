# 📊 Training

## Prerequisites

[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) 

```bash
pip install efficientnet_pytorch
```

### RGB-D Backbone

Generate training and validation splits for the RGB-D backbone. The splits will be saved in `data/creste/splits`.
```bash
python scripts/preprocessing/build_splits.py --split_type standard --cfg_file configs/dataset/distillation/creste_pefree_dinov2.yaml --out_dir data/creste/splits --horizon 50 --hausdorff 0 --min_distance 0
```

Run the following command to train the RGB-D backbone model. The model checkpoints will be saved under `model_ckpts/Dinov2Distillation`.

```bash
python creste/train_pefree.py 'dataset=distillation/creste_pefree_dinov2' 'model=distillation/effnet_ds2_dinov2_128' 'trainer=standard' 'model.batch_size=16' '+wandb_name=creste_mini'
```

### Reward Function 

```bash
python scripts/preprocessing/build_splits.py --split_type standard --cfg_file configs/dataset/traversability/creste_sam2elevtraverse_horizon.yaml --out_dir data/creste/splits --horizon 50 --hausdorff 0.0 --min_distance 0.0 --split_type curvature --overlap 10
```