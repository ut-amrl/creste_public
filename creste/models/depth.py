import os

import torch
from torch import nn

from omegaconf import DictConfig

from creste.utils.depth_utils import (convert_to_metric_depth, bin_depths, convert_to_metric_depth_differentiable)
from creste.models.vision_encoder import VisionEncoder
from creste.models.blocks.conv import MultiLayerConv

from creste.utils.visualization import (save_depth_color_image)

DEBUG_DEPTH=False

# Define your model architecture
class DepthCompletion(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(DepthCompletion, self).__init__()
        
        # Define your layers here
        self.vision_cfg = model_cfg.vision_backbone
        self.depth_cfg = model_cfg.depth_head
        self.discretize_cfg = model_cfg.discretize
        self.return_feats = self.vision_cfg.return_feats

        self.vision_backbone = VisionEncoder(self.vision_cfg)
        self.depth_head = MultiLayerConv(self.depth_cfg)

        # Load weights
        if os.path.isfile(self.vision_cfg.weights_path):
            self.load_weights(self.vision_cfg.weights_path)

    def load_weights(self, weights_path):
        """
        Loads weights from a pretrained model
        """
        print(f"Loading weights from {weights_path}")

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(weights_path, weights_only=False)['state_dict']

        # Filter out unnecessary model key prefix from training on pytorch lightning
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items() if k.startswith('model.')}
        state_dict = {k.replace('depthcomp.', '', 1) if k.startswith('depthcomp.') else k: v for k, v in state_dict.items()}

        # Filter any keys that are not in model
        current_model_keys = set(self.state_dict().keys())
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in current_model_keys:
                filtered_state_dict[k] = v
            else:
                print(f"Key {k} not found in current model keys")
        print(f'Loading depth model weights. removed {len(state_dict) - len(filtered_state_dict)} keys from state_dict')

        # Load weights
        self.load_state_dict(filtered_state_dict, strict=True)

    @staticmethod
    def _convert_to_metric_depth(x, discretize_cfg, valid_thres=0.9):
        """
        Converts depth logits to metric depth
        Inputs:
            depth_logits - [B, D, H, W] tensor of depth logits
        Outputs:
            depth - [B, H, W] tensor of depth in meters
        """
        # TOOD: Support filtering by probability of depth being valid
        depth_bins = x.argmax(dim=1)

        depth = convert_to_metric_depth_differentiable(
            x, 
            discretize_cfg.mode, 
            discretize_cfg.depth_min, 
            discretize_cfg.depth_max,
            discretize_cfg.num_bins
        )

        # depth = convert_to_metric_depth(
        #     depth_bins, 
        #     self.discretize_cfg.mode, 
        #     self.discretize_cfg.depth_min, 
        #     self.discretize_cfg.depth_max,
        #     self.discretize_cfg.num_bins
        # )

        with torch.no_grad():
            if DEBUG_DEPTH:
                # Sanity check if bins -> depth -> bins is identity
                depth_bins_ = bin_depths(
                    depth, 
                    discretize_cfg.mode, 
                    discretize_cfg.depth_min, 
                    discretize_cfg.depth_max,
                    discretize_cfg.num_bins
                )
                print("% Depth bins and depth bins_ are equal:", torch.sum(depth_bins == depth_bins_) / torch.prod(torch.tensor(depth_bins.shape)))

        return depth / 1000, depth_bins # Convert to meters
    
    def forward(self, x):
        """
        Performs depth completion on sparse LiDAR + RGB input

        D - number of depth bins 

        Inputs:
            x - (B, 4, H, W) tensor of sparse LiDAR + RGB input
        Returns:
            y - (B, D, H, W) tensor of dense depth prediction
        """

        # Sanity Check Inputs
        # import pdb; pdb.set_trace()
        # import cv2
        # import numpy as np
        # rgb = x[2, :3, :, :].permute(1, 2, 0).detach().cpu().numpy()
        # depth = x[2, -1, :, :].detach().cpu().numpy()
        # cv2.imwrite("testdepth.png", depth.astype(np.uint16))
        # cv2.imwrite("testrgb.png", rgb*255)

        feats = self.vision_backbone(x)

        outputs = {}
        outputs['depth_preds_logits'] = self.depth_head(feats)
        B, Z, Hs, Ws = feats.shape
        
        outputs['depth_preds_metric'], outputs['depth_preds_bins']   = \
            self._convert_to_metric_depth( outputs['depth_preds_logits'], self.discretize_cfg )

        if self.return_feats:
            outputs['depth_preds_feats'] = feats

        if DEBUG_DEPTH:
            print("Debugging depth")
            B, C, H, W = x.shape
            # Downsample bilinear interpolation for rgb image
            rgb = x.view(B, C, H, W)[:, :3, :, :]
            rgb = nn.functional.interpolate(rgb, size=(Hs, Ws), mode="bilinear")
            depth_bins = outputs['depth_preds_bins']
            depth = outputs['depth_preds_metric']

            # Sanity check depth bins
            save_depth_color_image(
                rgb[0].permute(1, 2, 0).detach().cpu().numpy(),
                depth_bins[0, :, :].detach().cpu().numpy(),
                "testdepthbins.png"
            )

            # Sanity check depth metric
            save_depth_color_image(
                rgb[0].permute(1, 2, 0).detach().cpu().numpy(),
                depth[0, :, :].detach().cpu().numpy(),
                "testdepthmetric.png"
            )
            import pdb; pdb.set_trace()

        return outputs