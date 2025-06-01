import torch
from torch import nn


class MultiLayerConv(nn.Module):
    """
    Initializes a multi layerconvolutional neural network with ReLU activations
    """

    def __init__(self, model_cfg):
        super(MultiLayerConv, self).__init__()

        self.model_cfg = model_cfg
        self.kernels = model_cfg.kernels
        self.paddings = model_cfg.paddings
        self.dims = model_cfg.dims
        self.norm_type = model_cfg.norm_type
        self.stride = model_cfg.get('stride', [1]*len(self.kernels))

        m = nn.ModuleList()
        for i in range(len(self.kernels)):
            m.append(nn.Conv2d(
                self.dims[i], self.dims[i+1],
                self.kernels[i], padding=self.paddings[i], stride=self.stride[i]
            ))
            if self.norm_type == 'batch_norm':
                m.append(nn.BatchNorm2d(self.dims[i+1]))
            m.append(nn.ReLU())
        self.model = nn.Sequential(*m)

    def forward(self, x):
        return self.model(x)

# TODO: Deprecate this


class ConvEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(ConvEncoder, self).__init__()

        self.model_cfg = model_cfg
        dims = model_cfg['dims']
        kernels = model_cfg['kernels']
        paddings = model_cfg['paddings']
        norm = model_cfg['norm_type']

        m = nn.ModuleList()
        assert len(kernels) == len(paddings)
        for i in range(len(kernels)):
            m.append(nn.Conv2d(dims[i], dims[i+1],
                               kernel_size=kernels[i], padding=paddings[i]))
            if norm == 'batch_norm':
                m.append(nn.BatchNorm2d(dims[i+1]))
            m.append(nn.ReLU())
        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return self.convs(x)

# https://github.com/PingoLH/FCHarDNet/blob/master/ptsemseg/models/hardnet.py


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1,
                 bn=False, norm_type='batch_norm', relu=True, bias=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias=bias))

        if bn:
            if norm_type == 'batch_norm':
                self.add_module('norm', nn.BatchNorm2d(out_channels))
            elif norm_type == 'group_norm':
                self.add_module('norm', nn.GroupNorm(
                    num_groups=2, num_channels=out_channels))
            else:
                raise Exception('Unknown norm type:', norm_type)

        if relu:
            self.add_module('relu', nn.ReLU(inplace=True))

        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)


class MultiScaleFCN(nn.Module):
    def __init__(self, model_cfg):
        super(MultiScaleFCN, self).__init__()
        self.model_cfg = model_cfg
        self.prepool_cfg = model_cfg.prepool
        self.postpool_cfg = model_cfg.postpool
        self.skip_cfg = model_cfg.skip
        self.trunk_cfg = model_cfg.trunk

        # ConvLayer automatically preserves spatial dimension
        prepool = nn.ModuleList()
        for i in range(len(self.prepool_cfg.kernels)):
            prepool.append(ConvLayer(
                self.prepool_cfg.dims[i], self.prepool_cfg.dims[i+1],
                kernel=self.prepool_cfg.kernels[i], stride=self.prepool_cfg.stride[i],
                bn=True, norm_type=self.prepool_cfg.norm_type, relu=True, bias=False
            ))
        self.prepool = nn.Sequential(*prepool)

        skip = nn.ModuleList()
        for i in range(len(self.skip_cfg.kernels)):
            skip.append(ConvLayer(
                self.skip_cfg.dims[i], self.skip_cfg.dims[i+1],
                kernel=self.skip_cfg.kernels[i], stride=self.skip_cfg.stride[i],
                bn=True, norm_type=self.skip_cfg.norm_type, relu=True, bias=False
            ))
        self.skip = nn.Sequential(*skip)

        trunk = nn.ModuleList()
        trunk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(len(self.trunk_cfg.kernels)):
            trunk.append(ConvLayer(
                self.trunk_cfg.dims[i], self.trunk_cfg.dims[i+1],
                kernel=self.trunk_cfg.kernels[i]
            ))
            if self.trunk_cfg.norm_type == 'batch_norm':
                trunk.append(nn.BatchNorm2d(self.trunk_cfg.dims[i+1]))
            trunk.append(nn.ReLU(inplace=True))
        # Deconv by 2
        trunk.append(nn.Upsample(scale_factor=2,
                     mode='bilinear', align_corners=False))
        self.trunk = nn.Sequential(*trunk)

        postpool = nn.ModuleList()
        for i in range(len(self.postpool_cfg.kernels)):
            postpool.append(ConvLayer(
                self.postpool_cfg.dims[i], self.postpool_cfg.dims[i+1],
                kernel=self.postpool_cfg.kernels[i], stride=self.postpool_cfg.stride[i], bn=True, norm_type=self.postpool_cfg.norm_type, relu=True, bias=False
            ))
        self.postpool = nn.Sequential(*postpool)

        self.initialize_weights_with_xavier()

    def initialize_weights_with_xavier(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Expects input of shape [B, C, H, W]"""
        # Prepool
        x = self.prepool(x)
        # Skip
        skip = self.skip(x)
        # Trunk
        x = self.trunk(x)
        # Concat skip and trunk
        x = torch.cat([x, skip], dim=1)
        # Postpool
        x = self.postpool(x)

        return x
