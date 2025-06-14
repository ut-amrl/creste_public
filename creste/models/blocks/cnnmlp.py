import torch
import torch.nn as nn
import torch.nn.functional as F

from creste.models.blocks.conv import MultiLayerConv
from creste.utils.train_utils import prefix_dict

class MultiLayerPerceptron(nn.Module):
    def __init__(self, dims, **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        self.dims = dims
        for key, value in kwargs.items():
            setattr(self, key, value)

        m = nn.ModuleList()
        for i in range(len(self.dims)-1):
            m.append(nn.Linear(dims[i], dims[i+1]))
            m.append(nn.ReLU())
        self.model = nn.Sequential(*m)
    
    def forward(self, x):
        return self.model(x)

class CnnMLP(nn.Module):
    def __init__(self,
            **kwargs
        ):
        super(CnnMLP, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)        

        #1 Initialize conv feature extraction
        self.conv_layers = globals()[self.cnn_cfg["name"]](
            self.cnn_cfg["net_kwargs"]
        )

        #2 Flatten features for MLP
        self.flatten = nn.Flatten()

        #3 Pool features to predict actions
        self.mlp_head = globals()[self.mlp_cfg["name"]](
            **self.mlp_cfg["net_kwargs"]
        )
        
    def forward(self, inputs):
        # Aggregate output features in early fusion
        x = None
        for key in self.input_keys:
            if x is None:
                x = inputs[key]
            else:
                x = torch.cat((x, inputs[key]), dim=1)
        
        # Convolutional layers with ReLU activations
        x = self.conv_layers(x)

        # Flattening the output for the fully connected layers
        x = self.flatten(x)
        
        # Fully connected layers with ReLU activations + regression layer
        y = self.mlp_head(x)
        
        # Reshape to [B, Tout, 2]
        y = y.view(-1, self.out_horizon, 2)

        # Return the output
        out = [ dict(preds=y) ]

        assert(len(out) == len(self.output_prefix))
        ret = dict()
        for p, o in zip(self.output_prefix, out):
            ret.update(prefix_dict(p, o, seprator='_'))

        return ret
