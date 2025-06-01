
import os
import torch
from torch import nn
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from creste.models.blocks.conv import MultiLayerConv
from creste.models.depth import DepthCompletion

from creste.utils.feature_extractor import (ViTExtractor, extract_vit_features)

class FoundationBackbone(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.vision_cfg = model_cfg.vision_backbone
        self.depth_cfg = model_cfg.depth_head
        self.discretize_cfg = model_cfg.discretize
        self.multiview_distillation = self.model_cfg.get('multiview_distillation', False)
        assert self.model_cfg.get('vision_backbone', None) is not None, "Vision backbone not provided"

        self.vision_backbone = ViTExtractor(self.vision_cfg['name'], self.vision_cfg['backbone_cfgs']['stride'])
        self.depth_head = MultiLayerConv(self.depth_cfg)

        self.ckpt_path = self.model_cfg.get('ckpt_path', '')
        self.weights_path = self.model_cfg.get('weights_path', '')
        if self.ckpt_path == '':
            self.ckpt_path = self.vision_cfg.get('ckpt_path', '')
        if self.weights_path == '':
            self.weights_path = self.vision_cfg.get('weights_path', '')

        # Ensure that vision backbone is frozen
        print("Freezing vision backbone")
        for param in self.vision_backbone.model.parameters():
            param.requires_grad = False
        self.vision_backbone.model.eval()
        self.input_image_shape = self.vision_cfg['backbone_cfgs']['input_shape']
        self.output_image_shape = self.vision_cfg['backbone_cfgs']['output_shape']

        # Create transforms
        self.transform = T.Compose([
            T.Resize(self.input_image_shape, interpolation=InterpolationMode.BILINEAR),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        assert ~(os.path.isfile(self.ckpt_path) or os.path.isfile(self.weights_path)), "Please provide a valid ckpt or weights path. Not both."
        if os.path.isfile(self.weights_path):
            self.load_weights(self.weights_path)
        if os.path.isfile(self.ckpt_path):
            self.load_weights(self.ckpt_path)

    def load_weights(self, weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path)['state_dict']
 
        # Filter out unnecessary model key prefix from training on pytorch lightning
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items() if k.startswith('model.')}
        state_dict = {k.replace('depthcomp.', '', 1) if k.startswith('depthcomp.') else k: v for k, v in state_dict.items()}
        # state_dict = {k.replace('depthcomp.depthcomp.', 'depthcomp.', 1): v for k, v in state_dict.items()}

        # Remove any bevclassifier and cam2map
        state_dict = {k: v for k, v in state_dict.items() if 'bevclassifier' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'cam2map' not in k}

        self.load_state_dict(state_dict, strict=True)

        print(f'Loaded weights with strict mode successfully')

    def unfreeze_backbone(self):
        """
        Only unfreezes non dinov2 backbone
        """
        print("Unfreezing only depth head, not vision backbone")
        for param in self.depth_head.parameters():
            param.requires_grad = True

    def _preprocess(x):
        return self.transform(x)

    def forward(self, x):
        """
        Peforms depth completion using RGB input

        D - number of depth channels

        Inputs:
            x - (B, Tin, 3, H, W) tensor
            p2p - (B, Tin, 4, 4) pixel to point transformation
        Returns:
            y - (B, D, H, W) tensor of depth logits
        """
        if self.multiview_distillation:
            x = x[0] # Only use x, not p2p
        assert x.ndim == 5, f"Expected 4D input, got {x.ndim}"
        B, Tin, C, Hi, Wi = x.shape
        # Ensure that only the first 3 channels are used
        x = x.view(B*Tin, C, Hi, Wi) # Collapse batch and time dimensions
        # x = x[:, :3] # Only use the first 3 channels

        # Extract and resize features
        x = self.transform(x)
        feats = extract_vit_features(
            self.vision_backbone, x, self.input_image_shape)

        B, N, F = feats.shape
        H, W = self.vision_backbone.num_patches
        import pdb; pdb.set_trace()
        assert N == (H*W), f"Expected {H*W} features, got {N}" 
        feats = feats.view(B, H, W, F).permute(0, 3, 1, 2) # [B, F, H, W]
        feats = T.Resize(self.output_image_shape, interpolation=InterpolationMode.BILINEAR)(feats)
        
        outputs = {}
        outputs['depth_preds_feats'] = feats
        outputs['depth_preds_logits'] = self.depth_head(feats)
        B, Z, Hs, Ws = feats.shape
        outputs['depth_preds_metric'], outputs['depth_preds_bins'] = \
            DepthCompletion._convert_to_metric_depth( outputs['depth_preds_logits'], self.discretize_cfg )

        return outputs


if __name__ == '__main__':
    # Load the model config
    model_cfg = OmegaConf.load('configs/model/foundation/dinov2_rgbonly.yaml')

    # Create the model
    model = FoundationBackbone(model_cfg)
    model = model.cuda()

    # Load image, perform forward pass
    image_paths = [
        "/robodata/arthurz/Research/lift-splat-map/data/creste_rlang/2d_rect/cam0/16/2d_rect_cam0_16_700.jpg",
        "/robodata/arthurz/Research/lift-splat-map/data/creste_rlang/2d_rect/cam0/16/2d_rect_cam0_16_900.jpg"
    ]

    image_th = [T.ToTensor()(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda() for image_path in image_paths]
    image_th = torch.cat(image_th, dim=0) # [B, C, H, W]

    # Downsample input image by 2x
    image_th = nn.functional.interpolate(image_th, scale_factor=0.5, mode='bilinear', align_corners=False)
    image_th = image_th.unsqueeze(1) # [B, 1, C, H, W]

    import pdb; pdb.set_trace()
    y = model(image_th)
    
    for k, v in y.items():
        print(k, v.shape)