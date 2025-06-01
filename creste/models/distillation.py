import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from creste.models.blocks.conv import MultiLayerConv
from creste.models.stereodepth import MSNet2D
from creste.models.depth import DepthCompletion
from creste.models.blocks.splat_projection import Camera2MapMulti

from creste.utils.train_utils import (resize_and_center_crop)

class DistillationBackbone(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.vision_cfg = model_cfg.vision_backbone
        self.depth_cfg = model_cfg.depth_head
        self.distillation_cfg = model_cfg.distillation_head
        self.input_image_shape = self.vision_cfg['effnet_cfgs']['image_size']

        self.ckpt_path = self.model_cfg.get('ckpt_path', '')
        self.multiview_distillation = self.model_cfg.get('multiview_distillation', False)
        if self.ckpt_path == '':
            self.ckpt_path = self.vision_cfg.get('ckpt_path', '')
        self.weights_path = self.model_cfg.get('weights_path', '')
        if self.weights_path == '':
            self.weights_path = self.vision_cfg.get('weights_path', '')
        self.freeze_weights = model_cfg.get('freeze_weights', False)

        print("# ======== Initializing Depth Completion Head ======== #")
        print("Initializing DepthCompletion")
        self.depth_trunk = self.vision_cfg.get("depth_trunk", "DepthCompletion")
        try:
            self.depthcomp = globals()[self.depth_trunk](self.model_cfg)
        except: 
            print(f"Model {self.depth_trunk} not found")
            raise NotImplementedError

        if self.multiview_distillation:
            print("# ======== Initializing Splat Projection Head ======== #")
            self.camproj_cfg = model_cfg.get('camera_projector', None)
            assert self.camproj_cfg is not None, "Camera projector not provided for multiview distillation"
            print("Initializing Camera2MapMulti")
            self.cam2map = Camera2MapMulti(
                self.camproj_cfg,
                mode="bilinear",
                scatter_mode="max" # Max scatter works better
            )

        print("# ======== Initializing Feature Head ======== #")
        self.pe_map_cfg = model_cfg.get('pe_map', None)
        if self.pe_map_cfg is not None:
            print("Initializing PE Map")
            # globally-shared learnable PE map
            ds = self.vision_cfg["effnet_cfgs"]["downsample"]
            output_img_size = [img_dim // ds for img_dim in self.vision_cfg["effnet_cfgs"]["image_size"]]
            self.learnable_pe_map = nn.Parameter(
                0.05 * torch.randn(1, model_cfg.fdn_embed_dim//2, self.pe_map_cfg.height, self.pe_map_cfg.width),
                requires_grad=True,
            )
            # a PE head to decode learned PE map
            if self.pe_map_cfg.use_norm:
                self.pe_head = nn.Sequential(
                    nn.Conv2d(model_cfg.fdn_embed_dim//2, model_cfg.fdn_embed_dim, kernel_size=1, padding=0),
                    nn.BatchNorm2d(model_cfg.fdn_embed_dim),
                )
            else:
                self.pe_head = nn.Sequential(
                    nn.Conv2d(model_cfg.fdn_embed_dim//2, model_cfg.fdn_embed_dim, kernel_size=1, padding=0)
                )
        # Feature prediction head (Depth wise conv for efficiency)
        try: 
            dino_head_cfg = self.distillation_cfg.feature_head
            self.dino_head = globals()[dino_head_cfg.name](dino_head_cfg)
        except:
            print(f'Feature head {dino_head_cfg.name} not found')
            raise NotImplementedError
    
        # Load weights if provided and valid
        assert ~(os.path.isfile(self.ckpt_path) or os.path.isfile(self.weights_path)), "Please provide a valid ckpt or weights path. Not both.z"

        if os.path.isfile(self.ckpt_path):
            self.load_weights(self.ckpt_path)
        if os.path.isfile(self.weights_path):
            self.load_weights(self.weights_path)

    def load_weights(self, weights_path):
        """
        Loads weights from a checkpoint
        """
        print(f"Loading model weights from {weights_path}")
        
        state_dict = torch.load(weights_path)['state_dict']

        # TODO: Fix this hack for loading weights
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items() if k.startswith('model.')}
        state_dict = {k.replace('depthcomp.depthcomp.', 'depthcomp.', 1): v for k, v in state_dict.items()}
        state_dict = {k.replace('depthcomp.dino_head.', 'dino_head.', 1): v for k, v in state_dict.items()}
        
        # Remove any bevclassifier and cam2map
        state_dict = {k: v for k, v in state_dict.items() if 'bevclassifier' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'cam2map' not in k}

        # Load the state dict into the DepthCompletion model
        if self.freeze_weights:
            print("Freezing weights with freezing (strict)")
            self.load_state_dict(state_dict, strict=True)
            # Freeze only the parameters that are present in the saved_state_dict
            for name, param in self.named_parameters():
                if name in state_dict:
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # Optionally make sure other params are trainable
        else:
            print("Load all weights without freezing (strict)")
            # TODO remove this hack for learnable pe
            # print("Removing learnable_pe_map from state_dict, Remove this later!")
            # state_dict = {k: v for k, v in state_dict.items() if 'learnable_pe_map' not in k}
            # self.load_state_dict(state_dict, strict=False)
            self.load_state_dict(state_dict, strict=True)

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone for finetuning
        """
        print("Unfreezing rgbd backbone")
        for param in self.depthcomp.parameters():
            param.requires_grad = True

    def freeze_pe_map(self):
        """
        Freeze the learnable PE map and pe map head
        """
        self.learnable_pe_map.requires_grad = False
        for param in self.pe_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Tin     - number of observation frames
        Tout    - number of output frames

        Inputs:
            x: (torch.Tensor) [B, Tin, C, H, W] tensor of input features 
        Outputs:
            Dictionary with outputs from the backbone and BC head
            y: (torch.Tensor) [B, Tout, 2]
        """
        # Handle multiview and single view inputs
        if self.multiview_distillation:
            rgbd, p2p = x
            B, V, C, H, W = rgbd.shape
        else:
            rgbd = x
            B, V, C, H, W = rgbd.shape

        # Resize and center crop the input
        rgbd = rgbd.view(B*V, C, H, W)
        # rgbd = resize_and_center_crop([rgbd], [self.input_image_shape])[0]

        outputs = {}

        #1 Depth Completion
        outputs.update(self.depthcomp(rgbd))
        _, Z, Hs, Ws = outputs['depth_preds_feats'].shape
        # Hacky fix, figure out better way to handle multiple views and multiples frames
        V = 1
        depth   = outputs['depth_preds_metric'].view(B, V, Hs, Ws)
        feats   = outputs['depth_preds_feats'].view(B, V, Z, Hs, Ws)

        #2 PE Free Dino Feature Prediction Head
        dino_feats = self.dino_head(feats.view(B*V, Z, Hs, Ws))
        _, D, _, _ = dino_feats.shape

        if self.pe_map_cfg is not None:
            #2 PE Prediction Head (Does batch norm as well)
            learnable_pe_map = (
                F.interpolate(
                    self.learnable_pe_map,
                    size=(Hs, Ws),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            dino_pe = self.pe_head(learnable_pe_map)
            outputs['dino_pe'] = dino_pe

            #4 Add PE to PE Free Dino Feature
            dino_pe_feats = dino_feats + dino_pe
            outputs['dino_pefree_feats'] = dino_feats.view(B, V, D, Hs, Ws)
            outputs['dino_pe_feats'] = dino_pe_feats.view(B, V, D, Hs, Ws)

            if self.camproj_cfg is not None and self.multiview_distillation:
                #3 Differentiable Splat Projection
                outputs.update(self.cam2map((
                    depth, dino_feats.view(B*V, 1, D, Hs, Ws), p2p
                )))
        else:
            outputs['dino_pe_feats'] = dino_feats.view(B, V, D, Hs, Ws)
        
        return outputs