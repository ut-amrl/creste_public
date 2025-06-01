import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Efficient Net variations
from creste.models.blocks.effnet import EffNet

class VisionEncoder(nn.Module):
    """
    Superclass for RGB encoders. User selectable with rgb encoder to use

    """
    def __init__(self, vision_cfg):
        super(VisionEncoder, self).__init__()
        self.vision_cfg = vision_cfg
        self.input_type = vision_cfg.input_type
        self.name       = vision_cfg.name

        # Currently handle only RGB and RGBD inputs
        if self.input_type=="rgb" or self.input_type=="rgbd":
            if "efficientnet" in self.name:
                self.model = EffNet(
                    name=self.name,
                    inC=self.vision_cfg.effnet_cfgs.in_channels,
                    outC=self.vision_cfg.effnet_cfgs.out_channels,
                    image_size=self.vision_cfg.effnet_cfgs.image_size,
                    downsample=self.vision_cfg.effnet_cfgs.downsample,
                    return_2nd_last_layer_output=False # Returnly feature map
                )
        else:
            raise NotImplementedError(f"Input type {self.input_type} not supported")

    def forward(self, img):
        """
        x - [B, N, C, H, W] RGB image in temporal order
                B: Batch size
                N: Number of frames
                C: Inputs channels (3 for RGB, 4 for RGBD)
                H: Height
                W: Width
        """
        B, C, H, W = img.shape

        if self.input_type=="rgb":
            img = img[:, :3, :, :]
        # Forward pass      
        output = self.model(img) # [B, C, H, W]

        return output

if __name__ == '__main__':
    print("------   Demo of RGBD feature maps   -----") 

    # Create a dummy input
    pass