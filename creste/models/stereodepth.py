"""
Original Implementation of mobilestereonet
https://github.com/cogsys-tuebingen/mobilestereonet
"""
from __future__ import print_function
import os
import math
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from creste.models.blocks.conv import MultiLayerConv
from creste.models.vision_encoder import VisionEncoder
from creste.models.depth import DepthCompletion

from omegaconf import DictConfig
from creste.models.blocks.stereo_submodule import (feature_extraction, MobileV2_Residual, convbn, interweave_tensors)

class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=(1,0), stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=(1, 0), stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
 
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class HourGlassTrunk(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(HourGlassTrunk, self).__init__()
        self.hg_cfg = model_cfg.hg_cfgs
        self.preconv_cfg = model_cfg.preconv_cfgs

        self.num_groups = self.hg_cfg.num_groups
        self.volume_size = self.hg_cfg.volume_size
        self.hg_size = self.hg_cfg.hg_size
        self.dres_expanse_ratio = self.hg_cfg.dres_expanse_ratio

        self.preconv11 = MultiLayerConv(self.preconv_cfg)
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU())

        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.dres0 = nn.Sequential(
            MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
            nn.ReLU(inplace=True),
            MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
            nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(
            MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
            nn.ReLU(inplace=True),
            MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass2D(self.hg_size)

        self.encoder_decoder2 = hourglass2D(self.hg_size)

        self.encoder_decoder3 = hourglass2D(self.hg_size)

        self.classif0 = nn.Sequential(
            convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
            bias=False, dilation=1))

        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))

    def forward(self, x):
        featL, featR = x

        # Channel reduction for reduced memory footprint
        featL = self.preconv11(featL) # [B, Z, Hs, Ws] -> [B, Zs, Hs, Ws]
        featR = self.preconv11(featR) # [B, Z, Hs, Ws] -> [B, Zs, Hs, Ws]

        # Interlaced cost volume
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)

        cost3 = self.classif3(out3)
        cost3 = torch.unsqueeze(cost3, 1)
        # Feature interpolation redundant for latent disparity
        # cost3 = F.interpolate(cost3, [self.maxdisp, featL.size()[2], featL.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        # import pdb; pdb.set_trace()
        return {
            "cost3": cost3
        }



# Modify this to be encapsulated in distillation backbone
class MSNet2D(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(MSNet2D, self).__init__()
        self.maxdisp = model_cfg.num_depth_bins
        
        self.model_cfg = model_cfg
        self.vision_cfg = model_cfg.vision_backbone
        self.depth_cfg = model_cfg.depth_head
        self.discretize_cfg = model_cfg.discretize
        self.costvolume_cfg = model_cfg.costvolume_trunk
        self.return_feats = self.vision_cfg.return_feats

        # 1 Initialize vision backbone, depth, and semantic heads
        assert self.model_cfg.cams == 2, "Stereo depth network requires 2 cameras"
        self.vision_backbone = VisionEncoder(self.vision_cfg)
        self.depth_head = MultiLayerConv(self.depth_cfg)

        #2 Initialize cost volume trunk
        self.hourglass_trunk = HourGlassTrunk(self.costvolume_cfg)

        #3 This way of initailizing weights is outdated
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Conv3d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

        # Load weights
        if os.path.isfile(self.vision_cfg.weights_path):
            self.load_weights(self.vision_cfg.weights_path)

    def load_weights(self, weights_path):
        """
        Loads weights from a pretrained model
        """
        print(f"Loading weights from {weights_path}")

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(weights_path)['state_dict']

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

    def forward(self, x):
        """
        Performs stereo depth estimation on a pair of stereo images

        Inputs:
            x - (B, 2, 3, H, W) tensor of stereo image pairs
        Returns:
            y - (B, D, H, W) tensor of estimated disparity maps
            y_cont (B, 1, H, W) tensor of estimated continuous disparity maps
        """
        Bn, C, H, W = x.shape
        B, N = Bn // 2, 2
        assert (Bn % 2) == 0, "Stereo depth network requires 2 cameras"

        # x = x.view(B*N, C, H, W) # (B*2, 3, H, W)
        features = self.vision_backbone(x) # (B*2, Z, Hs, Ws)
        features_L = features[0::2]   # picks indices 0, 2, 4, ... => left images
        features_R = features[1::2]  # picks indices 1, 3, 5, ... => right images

        hourglass_outputs = self.hourglass_trunk( (features_L, features_R) ) # [B, 1, Zd, Hs, Ws]

        # TODO: test if iterative hierarchical cost volume refinement is necessary
        outputs = {}
        
        # Learn function for latent depth estimation
        outputs['depth_preds_logits'] = self.depth_head(hourglass_outputs['cost3'])
        outputs['depth_preds_metric'], outputs['depth_preds_bins'] = \
            DepthCompletion._convert_to_metric_depth( outputs['depth_preds_logits'], self.discretize_cfg )

        if self.return_feats:
            outputs['depth_preds_feats'] = features_L # Use left image features for output

        # pred3 = F.softmax(cost3, dim=1)
        # pred3 = disparity_regression(pred3, self.maxdisp)

        return outputs