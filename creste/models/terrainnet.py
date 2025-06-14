import os
import torch
import torch.nn as nn

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from creste.models.depth import DepthCompletion
from creste.models.distillation import DistillationBackbone
from creste.models.foundation import FoundationBackbone
from creste.models.blocks.splat_projection import Camera2MapMulti
from creste.models.blocks.rnn import MergeUnit
from creste.models.blocks.inpainting import InpaintingResNet18MultiHead
from creste.models.blocks.cnnmlp import MultiLayerPerceptron

from creste.utils.depth_utils import convert_to_metric_depth, bin_depths
from creste.utils.visualization import save_depth_color_image

import torch.nn.functional as F

DEBUG_TERRAINNET=False
SAVE_VISUALS=False

class TerrainNet(nn.Module):
    """
    SSC model for removal of objects and completion of scenes
    """
    def __init__(self, 
            model_cfg: DictConfig,
        ):
        super().__init__()
        self.model_cfg = model_cfg
        self.views = model_cfg.get('views', 1)

        # Vision encoders configs
        self.vision_cfg     = model_cfg.vision_backbone
        self.camproj_cfg    = model_cfg.camera_projector
        self.depth_cfg      = model_cfg.depth_head
        self.discretize_cfg = model_cfg.discretize
        self.projector_cfg  = model_cfg.get('projection_head', None)

        # Temporal Aggregation configs
        self.ckpt_path      = model_cfg.get('ckpt_path', '')
        self.weights_path   = model_cfg.get('weights_path', '')
        self.freeze_weights = model_cfg.get('freeze_weights', False)
        self.use_temporal   = model_cfg.get('use_temporal', False)
        self.use_movability = model_cfg.get('use_movability', False)
        self.load_setting   = model_cfg.get('load_setting', "strict")
        self.drop_decoder   = model_cfg.get('drop_decoder', False)

        # Inpainting configs
        self.bev_classifer_cfg = model_cfg.get('bev_classifier', None)
        if self.bev_classifer_cfg is not None:
            self.bev_classifer_cfg = OmegaConf.to_object(model_cfg.bev_classifier)
        
        self.bev_semantic_head_cfg  = model_cfg.get('bev_semantic_head', None)
        if self.bev_semantic_head_cfg is not None:
            self.bev_semantic_head_cfg = OmegaConf.to_object(self.bev_semantic_head_cfg)

        # Initialize rgbd backbone
        try:
            depthcomp_name = self.vision_cfg.get('class_name', None)
            if depthcomp_name is None:
                print("No depthcomp class name provided. Defaulting to DistillationBackbone...\n")
                depthcomp_name = "DistillationBackbone"
            print(f"Initializing {depthcomp_name}\n")
            self.depthcomp = globals()[depthcomp_name](
                self.model_cfg
            )
        except KeyError:
            raise NotImplementedError(f"Vision backbone {self.vision_cfg['class_name']} not implemented")

        # Projection and splat layer
        self.cam2map = Camera2MapMulti(
            self.camproj_cfg,
            mode="bilinear"
        )
        self.splat_key = self.camproj_cfg.get('splat_key', 'depth_preds_feats')
        print(f"TerrainNet() Initialization: Splatting key: {self.splat_key}")

        if self.use_temporal:
            self.temporal_cfg   = OmegaConf.to_object(model_cfg.temporal_layer)
            # Temporal Aggregation layer
            self.temporal_layer = MergeUnit(
                **self.temporal_cfg['net_kwargs']
            )

        # Inpainting head
        try:
            self.bevclassifier = None
            if self.bev_classifer_cfg is not None:
                self.bevclassifier = globals()[self.bev_classifer_cfg['name']](
                    **self.bev_classifer_cfg['net_kwargs']
                )
        except KeyError:
            raise NotImplementedError(f"Bev classifier {self.bev_classifer_cfg['name']} not implemented")

        # Semantic segmentation head
        try:
            if self.bev_semantic_head_cfg is not None:
                self.bev_semantic_head = globals()[self.bev_semantic_head_cfg['name']](
                    **self.bev_semantic_head_cfg['net_kwargs']
                )
        except KeyError:
            raise NotImplementedError(f"Semantic head {self.bev_semantic_head_cfg['name']} not implemented")

        # Load weights for terrainnet only if checkpoint not specified
        if os.path.isfile(self.weights_path) and not os.path.isfile(self.ckpt_path):
            self.load_weights(self.weights_path)

    def load_weights(self, weights_path):
        """
        Loads weights from a checkpoint
        """
        print(f"Loading terrainnet weights from {weights_path}")
        state_dict = torch.load(weights_path, weights_only=False)['state_dict']
        # Filter out unnecessary model key prefix from training on pytorch lightning
        state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in state_dict.items()}
        
        num_state_keys = len(state_dict.keys())
        new_state_dict = {}
        num_state_keys_changed = 0

        # This is a hacky way to load weights from a checkpoint that was trained with a different model
        if self.vision_cfg['class_name'] != "FoundationBackbone":
            for k, v in state_dict.items():
                # If key starts with depthcomp. but not depthcomp.depthcomp.
                # then insert the second depthcomp.
                if k.startswith("depthcomp.") and not k.startswith("depthcomp.depthcomp.") and not k.startswith("depthcomp.dino_head."):
                    new_k = k.replace("depthcomp.", "depthcomp.depthcomp.", 1)
                    print(f"Changing key: {k} -> {new_k}")
                    num_state_keys_changed += 1
                elif k.startswith("dino_head."):
                    new_k = k.replace("dino_head.", "depthcomp.dino_head.", 1)
                    print(f"Changing key: {k} -> {new_k}")
                    num_state_keys_changed += 1
                else:
                    new_k = k
                new_state_dict[new_k] = v
        else:
            new_state_dict = state_dict
        print(f"Number of keys changed: {num_state_keys_changed}")

        # 3) Ensure we have preserved the same number of keys
        assert num_state_keys == len(new_state_dict), (
            f"Number of keys changed after filtering. "
            f"Before: {num_state_keys}, After: {len(new_state_dict)}"
        )
        state_dict = new_state_dict

        # Load the state dict into the DepthCompletion model
        if self.load_setting=="ft_semantic_head":
            print("Loading weights for finetuning semantic head")

            #1 Load weights for full model, skip semantic head
            self.load_state_dict(state_dict, strict=False)

            #2 Freeze all weights except non semantic decoders
            for name, param in self.named_parameters():
                if 'bev_semantic_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            #3 Unfreeze weights for elevation decoder head
            for decoder_idx in range(len(self.bevclassifier.out_heads)):
                if self.bevclassifier.out_heads[decoder_idx].proj.out_channels==1:
                    for name, param in self.bevclassifier.out_heads[decoder_idx].named_parameters():
                        param.requires_grad = True

            if DEBUG_TERRAINNET:
                print("|---- No grad required for ----|\n")
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        print(name)
                print("|---- Grad required for ----|\n")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(name)
                import pdb; pdb.set_trace()
        elif self.load_setting == "ft_decoders_all":
            print("Loading weights for all but decoder heads")
            
            #1 Remove decoder head weights from state_dict
            print('Dropping decoder head weights with prefix bevclassifier.out_heads')
            state_dict = {k: v for k, v in state_dict.items() if 'bevclassifier.out_heads' not in k}

            #2 Load weights for full model, skip decoder heads
            self.load_state_dict(state_dict, strict=False)

            #3 Freeze all weights except decoder heads
            for name, param in self.named_parameters():
                if 'bevclassifier.out_heads' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if DEBUG_TERRAINNET:
                print("|---- No grad required for ----|\n")
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        print(name)
                print("|---- Grad required for ----|\n")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(name)
                import pdb; pdb.set_trace()
        elif self.load_setting == "ft_decoders_partial":
            print("Loading weights for all but last few layers of decoder heads")

            #1 Load all weights except last up2 and proj layers of decoder heads
            is_up2_out_head = lambda x: 'bevclassifier.out_heads' in x and 'up2' in x
            is_proj_out_head = lambda x: 'bevclassifier.out_heads' in x and 'proj' in x
            state_dict = {k: v for k, v in state_dict.items() if not is_up2_out_head(k) and not is_proj_out_head(k)}

            #2 Load weights for full model, skip decoder heads
            self.load_state_dict(state_dict, strict=False)

            #3 Freeze all weights except last up2 and proj layers of decoder heads
            for name, param in self.named_parameters():
                if is_up2_out_head(name) or is_proj_out_head(name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
            if DEBUG_TERRAINNET:
                print("|---- No grad required for ----|\n")
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        print(name)
                print("|---- Grad required for ----|\n")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(name)
                import pdb; pdb.set_trace()
        elif self.load_setting == "strict_freeze":
            print("Loading all weights (strict) and freezing")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}

            self.load_state_dict(state_dict, strict=True)
            for name, param in self.named_parameters():
                param.requires_grad = False
        elif self.load_setting == "strict":
            # Drop loss weights if not training
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
            print("Load all weights (strict)")       
            self.load_state_dict(state_dict, strict=True)
        elif self.load_setting == "strict_unfreezesplat":
            # Freeze RGB-D backbone and unfreeze splat layer
            print("Loading all weights (strict) and unfreezing splat layer")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
            self.load_state_dict(state_dict, strict=False)

            # Freeze all parameters except those with "cam2map" in their name
            for name, param in self.named_parameters():
                if "cam2map." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise ValueError(f"Invalid load_setting {self.load_setting}")

        # if self.freeze_weights:
        #     print("Freezing loaded weights")
        #     # Freeze only the parameters that are present in the saved_state_dict
        #     for name, param in self.named_parameters():
        #         if name in state_dict:
        #             param.requires_grad = False
        #         else:
        #             param.requires_grad = True  # Optionally make sure other params are trainable

    def forward(self, x):
        """
        x is a tuple containing the following

        x[0] - [B, N, 4, H, W] RGB image in temporal order
                B: Batch size
                N: Number of frames x Number of Images of same scene (interleaved)
                4: RGBD channels, 0:3 RGB, 3: Depth
                H: Height
                W: Width
        x[1] - [B, N, 4, 4] Pose matrix relative to first frame in sequence
        x[2] - [B, 4, 4] Camera to LiDAR frame projection matrix (homogeneous)
        """
        rgbd, p2p = x[:2]
        B, N, C, H, W = rgbd.shape

        outputs = {}
        #1 Encode rgb feats 
        outputs.update(self.depthcomp(rgbd.view(B, N, C, H, W)))
        
        assert self.splat_key in outputs, f"Expected {self.splat_key} in depthcomp outputs"
        Z, Hs, Ws    = outputs[self.splat_key].shape[-3:]

        # TODO: hacky fix to handle multi view inputs, single view outputs
        N = self.views
        depth           = outputs['depth_preds_metric'].view(B, N, Hs, Ws)
        feats           = outputs[self.splat_key].view(B, N, Z, Hs, Ws) # [B, N, F, H, W]

        if SAVE_VISUALS:
            # Save torch tensors to disk
            torch.save(outputs['depth_preds_metric'], 'depth_preds_metric.pt')
            torch.save(outputs['depth_preds_feats'], 'depth_preds_feats.pt')
            torch.save(p2p, 'p2p.pt')
            # SAve rgbd image to disk
            torch.save(rgbd, 'rgbd.pt')
            import pdb; pdb.set_trace()

        #2 Splat RGB features to ego stable BEV map
        if self.training and self.use_movability:
            #2a Double forward pass for training with anchor vs multiview
            outputs.update(self.cam2map([
                depth[:, 0:1, :, :], 
                feats[:, 0:1, :, :], 
                p2p[:, 0:1, :, :]
            ])) # [B*NS, C, H, W]
            if len(x)>2 and x[2] is not None:
                self.cam2map.NC = N
                outputs.update(self.cam2map([depth, feats, p2p, x[2]]) ) # [B*NS, C, H, W]
                self.cam2map.NC = 1
        else:
            outputs.update(self.cam2map([depth, feats, p2p]))

        BNS, Z, Hg, Wg = outputs['bev_features'].shape
        NS = BNS // B

        # Temporal Aggregation
        if self.use_temporal:
            # Initialize beginnign of sequence at beginning for each batch
            bos = torch.zeros((B, NS), dtype=torch.bool, device=rgbd.device)
            bos[:, 0] = True
            bos = bos.view(B*NS)
            gru_bev_features = self.temporal_layer(
                outputs['bev_features'], t=NS, bos=bos
            ).view(B, NS, Z, Hg, Wg) # [BNS, C, H, W] -> [B, N, C, H, W]

            # Extract last layer from ConvGRU
            outputs['merged_bev_features'] = gru_bev_features[:, -1, :, :, :]

        #3 Inpainting for single or multiview bev_features
        if self.bev_classifer_cfg is not None:
            if self.training and self.use_movability:
                outputs.update(self.bevclassifier(outputs)) # [B, C, H, W]
                outputs.update(self.bevclassifier(outputs, key_suffix="_mv")) # [B, C, H, W]
            else:
                outputs.update(self.bevclassifier(outputs)) # [B, C, H, W]

        if self.bev_semantic_head_cfg is not None:
            outputs.update(self.bev_semantic_head(outputs))
        return outputs
    
    