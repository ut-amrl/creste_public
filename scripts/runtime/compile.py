"""
This function compiles a pretrained model for deployment
"""
import os
import argparse
import yaml
import sys

# External imports
import cv2
import numpy as np
import pickle
import torch
import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F

# Local imports
from creste.datasets.coda_utils import SSC_LABEL_DIR, FSC_LABEL_DIR, TRAVERSE_LABEL_DIR, SAM_LABEL_DIR, SAM_DYNAMIC_LABEL_DIR
from creste.models.terrainnet import TerrainNet
from creste.models.lfd import MaxEntIRL as TraversabilityModel
from creste.utils.visualization import visualize_bev_label, visualize_elevation_3d_wrapper, draw_text_on_image
import creste.utils.train_utils as tu

CLI_ARGS=None

"""
Compiles traversability model for deployment
python runtime/compile.py 'model=traversability/inference/terrainnet_maxentirl_msfcn_sam2dynsemelev.yaml' \
    'model.weights_path=model_ckpts/TraversabilityLearning/depth128UD_dinov1pretrain_sam2dynelev_supcon_joint_BB_efficientnet-b0_Head_depthconv-head_lr_0.000500_UD_LAIDW_v2/creste_terrainnet_dinopretrain_maxentirl_msfcn_sam2dynsemelev_headMaxEntIRL_horizon50/20250115/092522/Adam-epoch\=42.ckpt' --model_type traversability --output runtime/traversability_model_trace.pt

python runtime/compile.py 'model=traversability/inference/terrainnet_maxentirl_msfcn_semelev' \
    'model.weights_path=runtime/dinov1pretrain_samelev_traversability_epoch49.ckpt'

python runtime/compile.py 'model=ssc_sam/terrainnet_supcon_sam2elev_dinov1pretrain' 'model.weights_path=model_ckpts/TerrainNetSAM/depth128UD_dinov1pretrain_sam2elev_supcon_joint_BB_efficientnet-b0_Head_depthconv-head_lr_0.000500_UD_IDW_v2/20241020/162104/Adam-epoch\=47.ckpt'
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile a pretrained model for deployment")
    parser.add_argument("--model_type", default="traversability", type=str,
                        help="Model to compile [traversability, bev_map]")
    parser.add_argument("--data_path", default="scripts/runtime/data_dict.pkl",
                        type=str, help="Path to the data dictionary")
    parser.add_argument("--output", type=str, default="scripts/runtime/creste_rgbd_trace.pt",
                        help="Path to save the compiled model")
    return parser.parse_known_args()


def visualize_model_output(inputs, outputs, path="runtime/output.png"):
    fov_mask = inputs['fov_mask'][0]
    fov_mask = fov_mask.cpu()

    # Visualize the output
    scene_image = np.empty((0, 256, 3), dtype=np.uint8)
    if '3d_ssc_label' in inputs.keys():
        features = torch.stack(
            [inputs['3d_ssc_label'], inputs['3d_ssc_label']], axis=0)
        prob = features / (torch.sum(features, dim=2, keepdim=True) + 1e-6)
        mode = torch.argmax(prob, dim=2)
        ssc_image = visualize_bev_label(SSC_LABEL_DIR, mode)
        H, W, _ = ssc_image.shape
        ssc_image = ssc_image[:H, :W//2]
        ssc_image[~fov_mask] = [0, 0, 0]
        scene_image = np.concatenate(
            [scene_image, ssc_image[:H//2, :]], axis=0)
    if '3d_sam_label' in inputs.keys():
        features = torch.stack(
            [inputs['3d_sam_label'], inputs['3d_sam_label']], axis=0)
        fsc_img = visualize_bev_label(SAM_LABEL_DIR, features)
        H, W, _ = fsc_img.shape
        fsc_img = fsc_img[:H, :W//2]
        fsc_img[~fov_mask] = [0, 0, 0]
        fsc_img = draw_text_on_image(fsc_img, 'GT', (10, 10))
        scene_image = np.concatenate(
            [scene_image, fsc_img[:H//2, :]], axis=0)
    if '3d_sam_dynamic_label' in inputs.keys():
        features = inputs['3d_sam_dynamic_label'][1:2]
        features = torch.stack([features, features], axis=0)
        sam_dynamic_img = visualize_bev_label(SAM_DYNAMIC_LABEL_DIR, features)
        H, W, _ = sam_dynamic_img.shape
        sam_dynamic_img = sam_dynamic_img[:H, :W//2]
        sam_dynamic_img[~fov_mask] = [0, 0, 0]
        sam_dynamic_img = draw_text_on_image(sam_dynamic_img, 'GT', (10, 10))
        scene_image = np.concatenate(
            [scene_image, sam_dynamic_img[:H//2, :]], axis=0)
    if 'inpainting_sam_preds' in outputs.keys():
        H, W = outputs['inpainting_sam_preds'].shape[-2:]
        preds = outputs['inpainting_sam_preds']
        pred_stack = torch.stack(
            [preds, preds], dim=0)
        pred_img = visualize_bev_label(FSC_LABEL_DIR, pred_stack, kwargs=None)
        pred_img = pred_img[:, :W]
        pred_img[~fov_mask] = [0, 0, 0]
        scene_image = np.concatenate([scene_image, pred_img[:H//2, :]], axis=0)
    if 'inpainting_sam_dynamic_preds' in outputs.keys():
        H, W = outputs['inpainting_sam_dynamic_preds'].shape[-2:]
        preds = outputs['inpainting_sam_dynamic_preds']
        if preds.shape[1] > 6:
            pred_stack = torch.stack(
                [preds, preds], dim=0)
            pred_img = visualize_bev_label(FSC_LABEL_DIR, pred_stack, kwargs=None)
        else:
            preds_mode = torch.argmax(preds, dim=1)
            pred_stack = torch.stack(
                [preds_mode, preds_mode], dim=0)
            pred_img = visualize_bev_label(SAM_DYNAMIC_LABEL_DIR, pred_stack, kwargs=None)
        pred_img = pred_img[:, :W]
        pred_img[~fov_mask] = [0, 0, 0]
        scene_image = np.concatenate([scene_image, pred_img[:H//2, :]], axis=0)
    if 'elevation_label' in inputs.keys():
        elevation_features = inputs['elevation_label'][:, 0]
        elevation_image = visualize_elevation_3d_wrapper(
            elevation_features, elevation_features,
            unoccluded_mask=fov_mask.unsqueeze(0)
        )
        H, W, _ = elevation_image.shape
        elevation_image = elevation_image[:H, W//2:]
        scene_image = np.concatenate(
            [scene_image, elevation_image[:, :W//2]], axis=0)
    if 'elevation_preds' in outputs.keys():
        preds = outputs['elevation_preds'][:, 0]
        elevation_image = visualize_elevation_3d_wrapper(
            preds, preds,
            unoccluded_mask=fov_mask.unsqueeze(0)
        )
        H, W, _ = elevation_image.shape
        scene_image = np.concatenate(
            [scene_image, elevation_image[:, :W//2]], axis=0)
    if 'traversability_preds' in outputs.keys():
        preds = outputs['traversability_preds']
        B, C, H, W = preds.shape
        label = torch.stack(
            [inputs['3d_sam_label'], inputs['3d_sam_label']], axis=0).byte()
        # prob = inputs['3d_ssc_label']
        # prob = prob / (torch.sum(prob, dim=1, keepdim=True) + 1e-6)
        # mode = torch.argmax(prob, dim=1).unsqueeze(1).byte()  # [B, 1, H, W]
        Ho, Wo = features.shape[-2:]

        crop_fov_mask = tu.resize_and_crop(
            fov_mask.unsqueeze(0).unsqueeze(
                0).byte(), (Ho//2, Wo//2), (0, H, 0, W)
        )
        label = tu.resize_and_crop(label, (Ho//2, Wo//2), (0, H, 0, W))
        traversability_image = visualize_bev_label(
            TRAVERSE_LABEL_DIR, [preds, label], kwargs={'label_type': SAM_LABEL_DIR, 'fov_mask': crop_fov_mask})

        scene_image = np.concatenate(
            [scene_image, traversability_image], axis=0)
    cv2.imwrite(path, scene_image)


def move_data_to_device(data, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    return data


@hydra.main(version_base=None, config_path="../../configs", config_name="traversability")
def main(cfg: DictConfig):
    # args, unknown = parse_args()
    model_type = CLI_ARGS.model_type
    data_path = CLI_ARGS.data_path
    output = CLI_ARGS.output

    # 1 Initialize model
    print("|--------    Loading Model Parameters    --------|")
    if model_type == "traversability":
        # Don't solve mdp for inference
        cfg['model']['solve_mdp'] = False
        model = TraversabilityModel(cfg['model'])
    elif model_type == "bev_map":
        model = TerrainNet(cfg['model'])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    # Only compute parameters in specific key in model
    # total_params = sum(p.numel() for p in model.backbone.parameters())

    # 2 Initialize data paths
    print("|--------    Loading Data Parameters    --------|")
    assert os.path.exists(data_path), "Data dictionary does not exist"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    # torch.save(data, "runtime/data_dict1_6675.pt")
    # Move data to device
    data = move_data_to_device(data, device)
    inputs = tuple([data['rgbd'].to(device), data['p2p'].to(device)])

    # Save depth label
    # torch.save(data['depth_label'], "depth_label.pt")
    # torch.save(data['p2p_in'], "p2p_in.pt")
    print("|--------    Compiling Model    --------|")
    # 3 Compile the model
    traced_script_module = torch.jit.trace(model, (inputs,), strict=False)

    print("|--------    Visaulize Dry Run Compiled Model    --------|")
    # 4 Dry run the compiled model
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data = move_data_to_device(data, device)
    inputs = tuple([data['rgbd'].to(device), data['p2p'].to(device)])
    outputs = traced_script_module(inputs)
    visualize_model_output(data, outputs, path="runtime/output.png")

    print("|--------    Saving Compiled Model    --------|")
    # 5 Save the compiled model
    traced_script_module.save(output)


if __name__ == "__main__":
    # Parse known and unknown arguments
    CLI_ARGS, unknown = parse_args()

    # Print argparse arguments
    print("Argparse Arguments:")
    print(vars(CLI_ARGS))

    # Pass unknown arguments to Hydra (these will be treated as overrides)
    sys.argv = [sys.argv[0]] + unknown
    main()
