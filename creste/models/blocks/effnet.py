import torch
from torch import nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet, utils


class Up(nn.Module):
    def __init__(self, inC, outC, scale_factor=2, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=False
        )

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1, bias=False),
            norm_layer(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=False),
            norm_layer(outC),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class EffNet(nn.Module):
    def __init__(self, name, inC, outC, image_size, downsample,
                 return_2nd_last_layer_output=True,
                 apply_final_batch_norm=False):
        super(EffNet, self).__init__()

        self.trunk = EfficientNet.from_pretrained(name)
        self.trunk.set_swish(memory_efficient=False)  # For export

        # orig_stem_w = self.trunk._conv_stem.weight
        conv2d = utils.get_same_padding_conv2d(image_size)
        if name == "efficientnet-b0":
            self.trunk._conv_stem = conv2d(
                inC, 32, kernel_size=3, stride=2, bias=False)
            channels = [320, 112, 40, 24, 16, inC]
        else:
            raise NotImplementedError
        # self.trunk._conv_stem.weight[:, :3] = orig_stem_w

        # Compute effective scaled image sizes
        scaled = None
        if image_size is not None:
            scaled = [tuple(image_size)]
            for i in range(5):
                scaled.insert(0, (scaled[0][0] // 2, scaled[0][1] // 2))
        # EfficientNet final feature map downsamples 32x
        # We want to upsample by scale so that
        # final downsample rate = downsample

        scale = 32 // downsample
        i = 0
        C = channels[0]
        while scale > 1:
            if scaled is None or not (scaled[i+1][0] % 2 or scaled[i+1][1] % 2):
                scale_factor = 2
            else:
                scale_factor = (scaled[i+1][0] / scaled[i]
                                [0], scaled[i+1][1] / scaled[i][1])
            scale = scale // 2
            i = i + 1
            C += channels[i]
            setattr(self, f'up{i}', Up(C, C, scale_factor))
        self.n_ups = i
        self.conv = nn.Conv2d(C, outC, kernel_size=1, padding=0)

        if apply_final_batch_norm:
            self.bn = nn.BatchNorm2d(outC)

        self.apply_final_batch_norm = apply_final_batch_norm
        self.return_2nd_last_layer_output = return_2nd_last_layer_output

    def forward(self, x):
        endpoints = self.trunk.extract_endpoints(x)
        endpoints['reduction_0'] = x
        n = 5
        x = endpoints[f'reduction_{n}']
        for i in range(1, self.n_ups + 1):
            x = getattr(self, f'up{i}')(x, endpoints[f'reduction_{n-i}'])
        y = self.conv(x)

        if self.apply_final_batch_norm:
            y = self.bn(y)
            y = F.relu(y, inplace=True)

        if self.return_2nd_last_layer_output:
            return y, x
        else:
            return y
