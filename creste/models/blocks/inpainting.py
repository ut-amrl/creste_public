import torch
import torch.nn as nn

import torchvision

from creste.models.blocks.effnet import Up
from creste.utils.train_utils import prefix_dict

class Inpainting(nn.Module):
    def __init__(self, input_key=None, output_prefix=None, learnable_loss_weight=False):
        """
        Args:
            input_key:
            output_prefix:
            learnable_loss_weight: if True the module will create a learnable parameter
                to learn the log variance of the corresponding loss.
                TODO: this is kind of a quick hack to implement learnable loss weight.
                      A more modular approach would be move the learnable loss weights
                      to somewhere else and save them into the weights file.
        """
        super(Inpainting, self).__init__()
        self.input_key = input_key or 'merged_bev_features'
        self.output_prefix = output_prefix or 'inpainting'
        if learnable_loss_weight:
            w = torch.tensor([0.0], dtype=torch.float32)
            self.log_var = nn.Parameter(w, requires_grad=True)
        else:
            self.log_var = None

    def forward(self, tensor_dict, key_suffix=""):
        x = tensor_dict[f'{self.input_key}{key_suffix}']
        out = self._forward(x)
        
        if self.log_var is not None:
            out['log_variance'] = self.log_var

        if isinstance(out, list):
            assert(isinstance(self.output_prefix, list))
            assert(len(out) == len(self.output_prefix))
            ret = dict()
            for p, o in zip(self.output_prefix, out):
                if p=="inpainting_sam":
                    p = f'{p}{key_suffix}'
                ret.update(prefix_dict(p, o, seprator='_'))
        else:
            assert(isinstance(out, dict))
            output_prefix = f'{self.output_prefix}{key_suffix}'
            ret = prefix_dict(output_prefix, out, seprator='_')

        return ret
    
class DeconvHead(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer):
        super(DeconvHead, self).__init__()
        self.up1 = Up(in_ch, 256, scale_factor=4, norm_layer=norm_layer)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True)
        )

        self.proj = nn.Conv2d(128, out_ch, kernel_size=1, padding=0)

    def forward(self, x1, x2):
        x = self.up1(x1, x2)
        x = self.up2(x)
        return self.proj(x), x

class InpaintingResNet18MultiHead(Inpainting):
    def __init__(self, num_input_features, num_classes,
                 norm_layer='batch_norm', **kwargs):
        super(InpaintingResNet18MultiHead, self).__init__(**kwargs)

        if norm_layer == 'batch_norm':
            norm_layer = nn.BatchNorm2d
        else:
            raise Exception('Unsupported norm layer:', norm_layer)

        trunk = torchvision.models.resnet.resnet18(pretrained=False, zero_init_residual=True,
                                                   norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(
            num_input_features, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.out_heads = nn.ModuleList()
        for n in num_classes:
            self.out_heads.append(DeconvHead(64 + 256, n, norm_layer))

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        ret = []
        for head in self.out_heads:
            pred, fea = head(x, x1)
            ret.append(dict(preds=pred, features=fea))
        return ret
