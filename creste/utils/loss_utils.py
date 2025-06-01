import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

import creste.utils.depth_utils as du
import creste.utils.utils as utils
from creste.models.losses.supcon_loss import MultiPosConLoss
from creste.models.losses.balancedsupcon_loss import BalContrastiveLoss
from creste.utils.visualization import visualize_bev_label
import creste.utils.train_utils as tu
from creste.datasets.coda_utils import (SSC_LABEL_DIR, SOC_LABEL_DIR, SAM_LABEL_DIR, FSC_LABEL_DIR)

from kornia.losses import focal_loss

# For maxentirl
from sklearn import neighbors

DEBUG_LOSS = 0
DEBUG_VICREG_LOSS = 0
DEBUG_MAXENT_LOSS = 0


class Loss(nn.Module):
    def __init__(self, name, config):
        super(Loss, self).__init__()

        self.config = config
        self._name = name + config.get('tag', '')
        self.weight = config.get('weight', 1.0)
        self.task = config.get('task', None)

    def forward(self, tensor_dict):
        loss_dict, meta_data = self.loss(tensor_dict)

        ret_loss_dict = dict()

        # Apply the learnable loss weights if present
        # Ref: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        logvar_key = self.config.get('logvar_key', None)
        if logvar_key is not None:
            log_var = tensor_dict[logvar_key]
            w = 1.0 / (2.0 * torch.exp(log_var))

            if logvar_key is not None:
                ret_loss_dict['log_std'] = (1.0, 0.5 * log_var)
        else:
            w = 1.0

        ret_loss_dict.update({k: (self.weight * w, v)
                             for k, v in loss_dict.items()})
        return ret_loss_dict, meta_data

    def loss(self, tensor_dict):
        raise Exception('Not Implemented!')

    @property
    def name(self):
        return self._name


class LossManager(nn.Module):
    def __init__(self, config):
        super(LossManager, self).__init__()

        self.config = config
        self.losses = nn.ModuleList()  # auotmatically moved to correct device
        for lc in config.loss:
            print(f'Adding loss {lc.name}')
            loss = self.get_loss(lc)
            self.losses.append(loss)

    def forward(self, tensor_dict):
        # Something weird is going on here with mp
        loss_dict, meta_data = {}, {}
        for loss in self.losses:
            loss_task = loss.task
            if loss_task is None or loss_task == tensor_dict['task']:
                ld, md = loss(tensor_dict)
                md = {f'{loss.name}/{key}': val for key, val in md.items()}
                ld = {f'{loss.name}/{key}': val for key, val in ld.items()}
                meta_data.update(md)
                loss_dict.update(ld)

        return loss_dict, meta_data

    def get_loss(self, config):
        g = self.config
        task = config.get('task', None)
        return globals()[config['name']](config)


class BalancedContrastiveLoss(Loss):
    def __init__(self, config):
        super(BalancedContrastiveLoss, self).__init__(config.name, config)

        self.views = config.get('views', 1)
        self.temperature = config.get('temperature', 0.4)
        self.pos_in_denom = config.get('pos_in_denom', False)
        self.log_first = config.get('log_first', True)
        self.a_lc = config.get('a_lc', 1.0)
        self.a_spread = config.get('a_spread', 1.0)
        self.lc_norm = config.get('lc_norm', False)
        self.use_labels = config.get('use_labels', True)
        self.clip_pos = config.get('clip_pos', 1.0)
        self.pos_in_denom_weight = config.get('pos_in_denom_weight', 1.0)

        self.supcon_loss = BalContrastiveLoss(
            views=self.views,
            type="l_spread",
            temp=self.temperature,
            pos_in_denom=self.pos_in_denom,
            log_first=self.log_first,
            a_lc=self.a_lc,
            a_spread=self.a_spread,
            lc_norm=self.lc_norm,
            use_labels=self.use_labels,
            clip_pos=self.clip_pos,
            pos_in_denom_weight=self.pos_in_denom_weight
        )

        self.ignore_index = config.get('ignore_index', 0)

        # Extracting relevant loss feature
        self.mask_key = config.get('mask_key', 'inputs/fov_mask')
        self.pred_key = config.get('pred_key', 'outputs/inpainting_preds')
        self.lab_key = config.get('lab_key', 'inputs/sem_label')
        self.task = config.get('task', '3d_ssc')

        # Limit maximum number of patches per class allowed to keep memory usage in check
        self.max_patches_per_class = config.get('max_patches_per_class', 150)

    def loss(self, tensor_dict):
        """
        If num contrastive views is > 1, latent features from augmented views
        are consecutive in batch dimension.

        Inputs:
            pred - [BV, Z, H, W] where Z is the latent feature dimension
            gt_label - [BV, C, H, W] where C is the class label index
            fov_mask - [BV, H, W] - fov_mask inputs
        """
        feats = tensor_dict[self.pred_key]
        gt_prob = tensor_dict[self.lab_key]
        fov_mask = tensor_dict[self.mask_key]

        C = gt_prob.shape[1]
        BV, Z, H, W = feats.shape
        B = BV // self.views

        # 1 Setup augmented views
        gt_label = torch.argmax(gt_prob, dim=1)        # [B, V, H, W]

        # 2 Zero out invalid labeled pixels and outside FOV mask
        valid_mask = fov_mask & (gt_label != self.ignore_index)

        # 3 Interpolate features to reduces dimensionality
        # Hs, Ws = H // 2, W // 2
        # feats = F.interpolate(
        #     feats, (Hs, Ws), mode='bilinear', align_corners=False
        # )
        # valid_mask = F.interpolate(
        #     valid_mask.float().unsqueeze(1), (Hs, Ws), mode='nearest-exact'
        # ).squeeze(1).bool()
        # gt_label = F.interpolate(
        #     gt_label.float().unsqueeze(1), (Hs, Ws), mode='nearest-exact'
        # ).squeeze(1).long()

        # 3 Extract min(k) counts from each label
        Hs, Ws = H, W
        # 4 Flatten and filter for loss
        valid_mask = valid_mask.view(B, self.views, Hs, Ws).permute(
            0, 2, 3, 1)  # [BV, H, W] -> [B, H, W, V]
        valid_mask = valid_mask[:, :, :, 0]  # [B, H, W]
        feats = feats.permute(0, 2, 3, 1).view(
            B, self.views, Hs, Ws, Z)  # [B, V, H, W, Z]
        feats = feats.permute(0, 2, 3, 1, 4).view(
            B, Hs, Ws, self.views, Z)  # [B, H, W, V, Z]
        feats = feats[valid_mask]  # [N, V, Z]

        gt_label = gt_label.view(B, self.views, Hs, Ws).permute(
            0, 2, 3, 1)  # [BV, H, W] -> [B, H, W, V]
        gt_label = gt_label[valid_mask][:, 0]  # [N]

        # Sample the median number of patches per class
        counts = torch.bincount(gt_label)
        non_zero_counts = counts[counts.nonzero(as_tuple=True)].float()
        mean_count = max(non_zero_counts.median().int(),
                         self.max_patches_per_class)
        selected_indices = tu.extract_max_per_class(
            gt_label, mean_count, return_indices=True)

        feats = feats[selected_indices, :, :]
        gt_label = gt_label[selected_indices]

        # 5 Compute loss using default eye mask (TODO: Change later if we apply non-constant augmentations)
        loss = self.supcon_loss(feats, gt_label)

        return {f'{self.task}/supcon/sem_loss': loss}, {}


class SupPixelConLoss(Loss):
    def __init__(self, config):
        super(SupPixelConLoss, self).__init__(config.name, config)

        self.views = config.get('views', 1)
        self.temperature = config.get('temperature', 0.1)
        self.epsilon_w = 1e-5
        if 'class_weights' in config:
            import numpy as np
            frequencies = np.loadtxt(config['class_weights'])
            self.register_buffer("class_weights",
                                 torch.from_numpy(
                                     1 / np.log(frequencies + self.epsilon_w)).float()
                                 )
            self.num_class = config['num_class']
            assert (self.num_class == len(self.class_weights))
        else:
            self.class_weights = None

        self.supcon_loss = MultiPosConLoss(
            temperature=self.temperature, class_weights=self.class_weights
        )

        self.ignore_index = config.get('ignore_index', -1)

        # Extracting relevant loss feature
        self.mask_key = config.get('mask_key', 'inputs/fov_mask')
        self.pred_key = config.get('pred_key', 'outputs/inpainting_preds')
        self.lab_key = config.get('lab_key', 'inputs/sem_label')
        self.lab_suffix_key = self.lab_key.split("/")[-1]
        self.task = config.get('task', '3d_ssc')

    def loss(self, tensor_dict):
        """
        Inputs:
            pred - [B, Z, H, W] where Z is the latent feature dimension
            gt_label - [B, C, H, W] where C is the class label index
            fov_mask - [B, H, W] - fov_mask inputs
        """
        preds = tensor_dict[self.pred_key]
        gt_prob = tensor_dict[self.lab_key]
        fov_mask = tensor_dict[self.mask_key]

        C = gt_prob.shape[1]
        BV, Z, H, W = preds.shape
        B = BV // self.views

        if C > 1:
            gt_label = torch.argmax(gt_prob, dim=1)  # [B*H*W]
        else:
            gt_label = gt_prob.squeeze(1)

        # Convert mask labels to be different per batch
        if self.lab_key == "inputs/3d_sam_label":
            gt_label = utils.remap_labels_in_batch(gt_label, ignore_idx=0)

        valid_mask = (gt_label != self.ignore_index) & fov_mask

        Hs, Ws = H, W
        # 1 Reduce to 1D predictions and labels
        valid_mask = valid_mask.view(B, self.views, Hs, Ws).permute(
            0, 2, 3, 1)  # [BV, H, W] -> [B, H, W, V]
        valid_mask = valid_mask[:, :, :, 0]  # [B, H, W]
        preds = preds.permute(0, 2, 3, 1).view(
            B, self.views, Hs, Ws, Z)  # [B, V, H, W, Z]
        preds = preds.permute(0, 2, 3, 1, 4).view(
            B, Hs, Ws, self.views, Z)  # [B, H, W, V, Z]
        preds = preds[valid_mask][:, 0]  # [N, Z]

        gt_label = gt_label.view(B, self.views, Hs, Ws).permute(
            0, 2, 3, 1)  # [BV, H, W] -> [B, H, W, V]
        gt_label = gt_label[valid_mask][:, 0]  # [N]

        counts = torch.bincount(gt_label)
        non_zero_counts = counts[counts.nonzero(as_tuple=True)].float()
        median_count = min(non_zero_counts.median().int(), 1000)
        selected_indices = tu.extract_max_per_class(
            gt_label, median_count, return_indices=True)

        preds = preds[selected_indices, :]
        gt_label = gt_label[selected_indices]
        loss_dict = self.supcon_loss({"feats": preds, "labels": gt_label})

        return {f'{self.task}/{self.lab_suffix_key}/supcon/sem_loss': loss_dict["loss"], f'{self.task}/{self.lab_suffix_key}/supcon/img_loss': loss_dict["image_loss"]}, {}


class FocalLoss(Loss):
    def __init__(self, config):
        super(FocalLoss, self).__init__(config.name, config)

        self.epsilon_w = 1e-5
        self.num_class = config['num_class']
        self.class_dim = config.get('class_dim', -1)
        if 'class_weights' in config:
            import numpy as np
            frequencies = np.loadtxt(config['class_weights'])
            self.register_buffer("class_weights",
                                 torch.from_numpy(
                                     1 / np.log(frequencies + self.epsilon_w)).float()
                                 )
            assert (self.num_class == len(self.class_weights))
        else:
            self.class_weights = None
        self.mask_key = config.get('mask_key', 'inputs/fov_mask')
        self.pred_key = config.get('pred_key', 'outputs/inpainting_preds')
        self.lab_key = config.get('lab_key', 'inputs/sem_label')
        self.ignore_index = config.get('ignore_index', None)
        self.task = config.get('task', '3d_ssc')
        self.loss_configs = {
            "alpha": config.get('alpha', 0.25),
            "gamma": config.get('gamma', 2.0),
            "reduction": "mean"
        }

    def loss(self, tensor_dict):
        """
        Inputs:
            [B, C, H, W] for both pred, gt_prob
            [B, H, W] - fov_mask inputs
        """
        # Convert class counts to probabilities
        pred = tensor_dict[self.pred_key]
        gt = tensor_dict[self.lab_key] # [B, 3, H, W]

        if self.class_dim < 0:
            gt_prob = gt / (torch.sum(
                gt, dim=1, keepdim=True
            ) + self.epsilon_w)
            # Compute hard loss to use ignore_index
            gt_mode = torch.argmax(gt_prob, dim=1)
        else:
            gt_mode = gt[:, self.class_dim, :, :].long() # [B, H, W] Class IDs

        fov_mask = tensor_dict[self.mask_key]
    
        if DEBUG_LOSS:
            with torch.no_grad():
                print("Debugging Focal loss")
                pred_prob = torch.softmax(pred, dim=1)
                visualize_bev_label(
                    torch.argmax(pred_prob, dim=1),
                    gt_mode,
                    filepath="bev_output.png"
                )
                print("Saved predictions to bev_output.png")

                import cv2
                import numpy as np
                test_fov_mask = fov_mask[0].detach().cpu().numpy()
                test_fov_mask = test_fov_mask.astype(np.uint8)
                cv2.imwrite("fov_mask.png", test_fov_mask * 255)

        # Mask all values that are not in accumulated bird eye view
        pred = pred.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        pred = pred[fov_mask, :]  # [N, C]

        gt_mode = gt_mode[fov_mask]  # [N]
        loss = focal_loss(pred, gt_mode, weight=self.class_weights, **self.loss_configs)

        # Compute mIoU Accuracy
        with torch.no_grad():
            # Convert prediction logits to probabilities
            pred_prob = torch.softmax(pred, dim=1)
            pred_mode = torch.argmax(pred_prob, dim=1)

            # Mask unlabeled target values
            valid_mask = gt_mode != self.ignore_index

            valid_pred_mode = pred_mode[valid_mask]
            valid_gt_mode = gt_mode[valid_mask]

            pred_acc = torch.sum(
                valid_pred_mode == valid_gt_mode) / torch.numel(valid_gt_mode)

        return {f'{self.task}/cls_loss': loss}, {f'{self.task}/FocalLoss/mIoU': pred_acc}

class CrossEntropy(Loss):
    def __init__(self, config):
        super(CrossEntropy, self).__init__(config.name, config)

        self.num_class = config['num_class']
        self.epsilon_w = 1e-5
        if 'class_weights' in config:
            import numpy as np
            frequencies = np.loadtxt(config['class_weights'])
            self.register_buffer("class_weights",
                                 torch.from_numpy(
                                     1 / np.log(frequencies + self.epsilon_w)).float()
                                 )
            assert (self.num_class == len(self.class_weights))
        else:
            self.class_weights = None

        self.mask_key = config.get('mask_key', 'inputs/fov_mask')
        self.pred_key = config.get('pred_key', 'outputs/inpainting_preds')
        self.lab_key = config.get('lab_key', 'inputs/sem_label')
        self.ignore_index = config.get('ignore_index', None)
        self.task = config.get('task', '3d_ssc')
        self.class_dim = config.get('class_dim', -1)

        if self.ignore_index is not None:
            self.cs_loss = torch.nn.CrossEntropyLoss(
                reduction='mean', ignore_index=self.ignore_index, weight=self.class_weights
            )
        else:
            self.cs_loss = torch.nn.CrossEntropyLoss(
                reduction='mean', weight=self.class_weights
            )

    def loss(self, tensor_dict):
        """
        Inputs:
            [B, C, H, W] for both pred, gt_prob
            [B, H, W] - fov_mask inputs
        """
        # Convert class counts to probabilities
        pred = tensor_dict[self.pred_key]
        gt = tensor_dict[self.lab_key] # [B, F, H, W]
        if self.class_dim < 0:
            gt_prob = gt / (torch.sum(
                gt, dim=1, keepdim=True
            ) + self.epsilon_w)
            # Compute hard loss to use ignore_index
            gt_mode = torch.argmax(gt_prob, dim=1)
        else:
            gt_mode = gt[:, self.class_dim, :, :].long() # [B, H, W] Class IDs
        fov_mask = tensor_dict[self.mask_key]

        if DEBUG_LOSS:
            with torch.no_grad():
                print("Debugging CrossEntropy loss")
                pred_prob = torch.softmax(pred, dim=1)
                inputs = torch.stack([
                    torch.argmax(pred_prob, dim=1),
                    torch.argmax(gt_prob, dim=1)
                ], dim=0)

                visualize_bev_label(
                    SSC_LABEL_DIR, inputs,
                    filepath="bev_output.png"
                )
                print("Saved predictions to bev_output.png")

                import cv2
                import numpy as np
                test_fov_mask = fov_mask[0].detach().cpu().numpy()
                test_fov_mask = test_fov_mask.astype(np.uint8)
                cv2.imwrite("fov_mask.png", test_fov_mask * 255)

        # Mask all values that are not in accumulated bird eye view
        pred = pred.permute(0, 2, 3, 1)  # [B, H, W, C]
        pred = pred[fov_mask, :]  # [N, C]
        gt_mode = gt_mode[fov_mask]  # [N]
        loss = self.cs_loss(pred, gt_mode)

        # Compute mIoU Accuracy
        with torch.no_grad():
            # Convert prediction logits to probabilities
            pred_prob = torch.softmax(pred, dim=1)
            pred_mode = torch.argmax(pred_prob, dim=1)

            # Mask unlabeled target values
            # valid_mask = (gt_mode != self.ignore_index)
            valid_mask = gt_mode != 0 # 0 is assummed to always be ignore index

            valid_pred_mode = pred_mode[valid_mask]
            valid_gt_mode = gt_mode[valid_mask]

            pred_acc = torch.sum(
                valid_pred_mode == valid_gt_mode) / (torch.numel(valid_gt_mode) + self.epsilon_w)

        return {f'{self.task}/cls_loss': loss}, {f'{self.task}/mIoU': pred_acc}


class CrossEntropyDepth(Loss):
    """
    A helper loss function for training depth prediction
    as a classification problem.
    """

    def __init__(self, config):
        super(CrossEntropyDepth, self).__init__(config.name, config)

    def loss(self, tensor_dict):
        """
        Inputs:
            pred_key: [B*S, H, W] tensor of predicted depth logits
                B - batch size, S - number of images for latest frame
            lab_key: [BNS, H, W] tensor of ground truth depth
                BNS - batch size x number of images for latest frame x number of frames
        """
        B, S, H, W = tensor_dict[self.config['lab_key']].shape
        BNS, C, H, W = tensor_dict[self.config['pred_key']].shape
        N = BNS // (B*S)

        pred = tensor_dict[self.config['pred_key']]
        gt = tensor_dict[self.config['lab_key']].view(B*S, H, W)

        if pred.shape[0] != gt.shape[0]:
            pred = pred.view(B*S, N, C, H, W)
            # This is for multi frame depth prediction
            # Only take depth predictions for last frame
            pred = pred[:, -1, :, :, :]
        elif pred.shape[-2:] != gt.shape[-2:]:
            # This is a workaround for low-resolution depth prediction
            gt = F.interpolate(gt, pred.shape[-2:], mode='nearest').detach()

        # Convert depth to bins
        gt_bin = du.bin_depths(depth_map=gt, target=True,
                               **self.config['discretize'])
        flat_gt = gt_bin.flatten(1, 2).long()
        valid_gt = flat_gt != self.config['discretize']['num_bins']

        # Compute loss
        flat_pred = pred.permute(0, 2, 3, 1).flatten(1, 2)
        loss = F.cross_entropy(flat_pred[valid_gt, :], flat_gt[valid_gt])

        # Compute mIoU Accuracy
        with torch.no_grad():
            pred_probs = torch.softmax(flat_pred[valid_gt, :], dim=1)
            cls_preds = torch.argmax(pred_probs, dim=1)
            pred_acc = torch.sum(
                flat_gt[valid_gt] == cls_preds) / torch.numel(flat_gt[valid_gt])

        return {'depth/cls_loss': loss}, {'depth/acc': pred_acc}


class SmoothL1Depth(Loss):
    def __init__(self, config):
        super(SmoothL1Depth, self).__init__(config.name, config)

        beta = config['beta']
        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']

        self.smoothl1_loss = torch.nn.SmoothL1Loss(beta=beta, reduction='mean')

    def loss(self, tensor_dict):
        """
        Inputs:
            pred_key: [B*S, H, W] tensor of predicted depth logits
                B - batch size, S - number of images for latest frame
            lab_key: [BNS, H, W] tensor of ground truth depth
                BNS - batch size x number of images for latest frame x number of frames
        """
        B, S, H, W = tensor_dict[self.config['lab_key']].shape
        BNS, H, W = tensor_dict[self.config['pred_key']].shape
        N = BNS // (B*S)

        pred = tensor_dict[self.config['pred_key']]
        gt = tensor_dict[self.config['lab_key']].view(B*S, H, W)

        if pred.shape[0] != gt.shape[0]:
            pred = pred.view(B*S, N, H, W)
            # This is for multi frame depth prediction
            pred = pred[:, -1, :, :]
        elif pred.shape[-2:] != gt.shape[-2:]:
            # This is a workaround for low-resolution depth prediction
            gt = F.interpolate(gt, pred.shape[-2:], mode='nearest').detach()
        
        # Convert depth to bins
        gt_bin = du.bin_depths(depth_map=gt, target=True,
                               **self.config['discretize'])
        valid_gt = gt_bin != self.config['discretize']['num_bins']

        # Compute loss
        gt_m = gt / 1000.0
        loss = self.smoothl1_loss(
            pred[valid_gt].float(), gt_m[valid_gt].float())

        return {'depth/reg_loss': loss}, {}


class SmoothL1(Loss):
    def __init__(self, config):
        super(SmoothL1, self).__init__(config.name, config)

        beta = config['beta']
        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.absolute = config.get('absolute', False)
        self.take_grad = config.get('take_grad', False)

        self.smoothl1_loss = torch.nn.SmoothL1Loss(beta=beta)

    def loss(self, tensor_dict):
        pred = tensor_dict[self.pred_key]
        gt = tensor_dict[self.lab_key]

        if not self.absolute: # Relative error for >1 channels
            gt[:, 1, :, : ] = gt[:, 1, :, : ] - gt[:, 0, :, : ]

        if self.take_grad:
            assert (len(pred.shape) == 4)  # BCHW
            pred = torch.cat(torch.gradient(pred, axis=[2, 3]), axis=1)
            gt = torch.cat(torch.gradient(gt, axis=[2, 3]), axis=1)

        valid = ~torch.isnan(gt) & ~torch.isinf(gt)
        loss = self.smoothl1_loss(pred[valid], gt[valid])

        return {'val': loss}, {}


class MSELoss(Loss):
    def __init__(self, config):
        super(MSELoss, self).__init__(config.name, config)

        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.coords_key = config.get('coords_key', "outputs/bev_coords")
        self.overlap_only = config.get('overlap_only', False)

        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def loss(self, tensor_dict):
        pred = tensor_dict[self.pred_key]
        gt = tensor_dict[self.lab_key]

        B, Vp1, Z, H, W = pred.shape
        if "dino_pe_feats" in self.pred_key:
            B, V, Z, H, W = gt.shape
            gt = gt.permute(0, 1, 3, 4, 2).reshape(B*V*H*W, Z)
            pred = pred.permute(0, 1, 3, 4, 2).reshape(B*V*H*W, Z)

            # # Normalize range pre gradient computation
            # with torch.no_grad():
            #     batch_mu = gt.mean(dim=0)
            #     batch_std = gt.std(dim=0)
            #     gt = F.batch_norm(
            #         gt, batch_mu, batch_std, None, None, False, 0.0, 1e-5
            #     )
            #     # gt = (gt - gt.min(dim=0)[0]) / (gt.max(dim=0)[0] - gt.min(dim=0)[0] + 1e-12)

        if self.overlap_only:
            bev_coords = tensor_dict[self.coords_key]
            img = tensor_dict['inputs/image'] if DEBUG_LOSS else None
            # Only compute mseloss for pixels with overlapping bev coords
            loss = tu.compute_pixel_bevoverlap_loss(
                bev_coords.view(B, Vp1, H, W, 2), pred, gt, img
            )
        else:
            valid = ~torch.isinf(gt)
            loss = self.mse_loss(pred[valid], gt[valid])

        return {'loss': loss}, {}


class PEFreeMSELoss(Loss):
    def __init__(self, config):
        super(PEFreeMSELoss, self).__init__(config.name, config)

        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.V = config['num_views']+1
        self.density_threshold = config.get("density_threshold", 1e-3)

        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def loss(self, tensor_dict):
        pred = tensor_dict[self.pred_key]
        density = tensor_dict[self.lab_key]

        # Treat first element in each view set as the anchor
        BV, Z, H, W = pred.shape

        assert BV % self.V == 0, "Invalid number of views/cams"
        B = BV // self.V

        pred = pred.view(B, self.V, Z, H, W)
        density = density.view(B, self.V, 1, H, W)

        anchor_views, overlap_views = pred[:, :1], pred[:, 1:]
        anchor_density, overlap_density = density[:, 0:1], density[:, 1:]
        # [B, 1, Z, H, W] -> [B, V-1, Z, H, W]
        anchor_views = anchor_views.repeat(1, self.V-1, 1, 1, 1)

        # Exclude the calculation of valid mask from gradient computation
        with torch.no_grad():
            # Compute the normalized log of the dot product of the densities
            log_density = torch.log(anchor_density * overlap_density + 1e-5)

            log_density -= log_density.min(1, keepdim=True)[0]
            log_density /= (log_density.max(1, keepdim=True)
                            [0] - log_density.min(1, keepdim=True)[0] + 1e-5)

            # # Soft feature overlap weighting using volume density
            # overlap_views   = overlap_views * log_density # [B, V, Z, H, W]
            # anchor_views     = anchor_views * log_density # [B, 1, Z, H, W] -> [B, V, Z, H, W]

            # Hard overlap filtering using volumetric density
            valid_mask = log_density > self.density_threshold

        if DEBUG_LOSS:
            with torch.no_grad():
                # Visualize log density as grayscale heatmap
                # [B, V, H, W, 1] -> [B, V, H, W]
                log_density = log_density.permute(0, 1, 3, 4, 2).squeeze()
                import cv2
                import numpy as np
                accum_img = np.zeros((H, W), dtype=np.uint16)
                for v in range(self.V-1):
                    log_density_img = log_density[0, v].detach().cpu().numpy()
                    # Convert to uint16 range
                    log_density_img = (log_density_img * 255).astype(np.uint8)
                    cv2.imwrite(f"debug/log_density.png", log_density_img)

                    overlap_density_img = overlap_density[0, v].squeeze(
                    ).detach().cpu().numpy()
                    overlap_density_img = (
                        overlap_density_img * 255).astype(np.uint8)
                    cv2.imwrite(f"debug/overlap_density.png",
                                overlap_density_img)

                    anchor_density_img = anchor_density[0].squeeze(
                    ).detach().cpu().numpy()
                    anchor_density_img = (
                        anchor_density_img * 255).astype(np.uint8)
                    cv2.imwrite(f"debug/anchor_density.png",
                                anchor_density_img)

        valid_mask = valid_mask.permute(0, 1, 3, 4, 2).reshape(-1,)
        overlap_views = overlap_views.permute(
            0, 1, 3, 4, 2).reshape(-1, Z)  # [B, V, H, W, Z] -> [B*V*H*W, Z]
        # [B, V, H, W, Z] -> [B*V*H*W, Z]
        anchor_views = anchor_views.permute(0, 1, 3, 4, 2).reshape(-1, Z)

        overlap_views = overlap_views[valid_mask]
        anchor_views = anchor_views[valid_mask]

        loss = self.mse_loss(anchor_views, overlap_views)

        return {'loss': loss}, {}


class VicregLoss(Loss):
    def __init__(self, config):
        super(VicregLoss, self).__init__(config.name, config)

        self.pred_key = config['pred_key']
        self.pred_mv_key = config['pred_mv_key']
        self.lab_key = config['lab_key']
        self.fov_key = 'inputs/fov_mask'

        self.sim_coeff = config.get('sim_coeff', 1.0)
        self.std_coeff = config.get('std_coeff', 1.0)
        self.cov_coeff = config.get('cov_coeff', 1.0)
        self.use_grid_sample = config.get('use_grid_sample', False)
        self.max_samples_per_label = config.get('max_samples_per_label', 2000)
        self.max_variance_samples = config.get('max_variance_samples', 1)

    def compute_pairwise_loss(self, preds, gt, mask):
        """
        Computes the pairwise mse loss between same patches in anchor and aggregated views

        Inputs:
            pred: [B, 2, Z, H, W] tensor of predicted single scan vs multi scan features
            gt: [B, Z, H, W] tensor of pseudo sam label masks
            mask: [B, H, W] tensor of fov and validity masks
        Outputs:
            mse_loss: [1] tensor of mse loss between same patches in anchor and aggregated views
        """
        B, _, Z, H, W = preds.shape

        total_mse_loss = 0
        count = 0
        for b in range(B):
            anchor_pred = preds[b, 0]
            mv_pred = preds[b, 1]
            mask_batch = mask[b]
            gt_batch = gt[b]

            if DEBUG_VICREG_LOSS:
                # Visualize pred features
                import cv2
                # combine mask to form (Hx2W)
                dual_mask = torch.cat(
                    [mask_batch, mask_batch], axis=1).unsqueeze(-1).cpu().numpy()
                feature_img = visualize_bev_label(
                    FSC_LABEL_DIR,
                    torch.stack([anchor_pred.unsqueeze(
                        0), mv_pred.unsqueeze(0)], axis=0)
                )
                label_img = visualize_bev_label(
                    SAM_LABEL_DIR,
                    torch.stack(
                        [gt_batch.unsqueeze(0), gt_batch.unsqueeze(0)], axis=0)
                )
                feature_img = feature_img * dual_mask
                label_img = label_img * dual_mask
                combined_img = np.concatenate((feature_img, label_img), axis=0)
                cv2.imwrite("test.png", combined_img)

            # Construct pairwise feature matrices
            # [B, Z, H, W] -> [B, H, W, Z] -> [N, Z]
            anchor_pred = anchor_pred.permute(1, 2, 0)[mask_batch].view(-1, Z)
            # [B, Z, H, W] -> [B, H, W, Z] -> [N, Z]
            mv_pred = mv_pred.permute(1, 2, 0)[mask_batch].view(-1, Z)
            gt_flat = gt_batch[mask_batch].view(-1, 1)  # [B, H, W] -> [N, 1]

            # Sample up to N patches for each label
            sampled_indices = torch.empty(
                0, dtype=torch.long, device=anchor_pred.device)
            unique_labels = torch.unique(gt_flat)
            for label in unique_labels:
                indices = torch.where(gt_flat == label)[0]

                if self.max_samples_per_label > 0:
                    if indices.shape[0] > self.max_samples_per_label:
                        indices = indices[torch.randperm(
                            indices.shape[0])[:self.max_samples_per_label]
                        ]

                sampled_indices = torch.cat((sampled_indices, indices))

            anchor_pred = anchor_pred[sampled_indices]
            mv_pred = mv_pred[sampled_indices]
            gt_flat = gt_flat[sampled_indices]
            N = anchor_pred.shape[0]

            anchor_exp = anchor_pred.unsqueeze(1).expand(-1, N, -1)
            mv_exp = mv_pred.unsqueeze(0).expand(N, -1, -1)
            mask_exp = (gt_flat == gt_flat.T)  # [N, 1] -> [N, N]

            # Compute pairwise log squared error
            mse = F.mse_loss(
                anchor_exp,
                mv_exp,
                reduction='none'
            )
            # Apply mask to MSE results
            mse_masked = mse * mask_exp.unsqueeze(-1)

            total_mse_loss += mse_masked.sum()
            count += mask_exp.sum()  # Number of matching label pairs

        mse_loss = total_mse_loss / count

        return mse_loss

    def off_diagonal(self, x):
        """
        Return a flattened view of the off-diagonal elements of a square matrix
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def compute_variance_loss(self, z1, gt):
        """
        Computes the variance across patches with dissimilar labels.

        Inputs:
            z1: [N, Z] tensor of features
            gt: [N, 1] tensor of pseudo sam label masks
        Outputs:
            std_loss: [1] tensor of variance loss across patches with dissimilar labels
        """
        # Sample a patch from each label
        unique_labels = torch.unique(gt)
        sampled_indices = torch.empty(0, dtype=torch.long, device=z1.device)
        for label in unique_labels:
            if label == 0:
                continue
            indices = torch.where(gt == label)[0]
            rand_indices = torch.randperm(indices.shape[0])[
                :self.max_variance_samples]

            sampled_indices = torch.cat(
                (sampled_indices, indices[rand_indices]))

        # Sampled features
        if sampled_indices.shape[0] > 0:
            sz1 = z1[sampled_indices]

            # Compute variance across dissimilar patches
            std_z1 = torch.sqrt(torch.var(sz1, dim=0) + 1e-4)
            std_loss = torch.mean(F.relu(1 - std_z1))

            return std_loss
        else:
            import pdb
            pdb.set_trace()
            return torch.tensor(0.0, device=z1.device)

    def loss(self, tensor_dict):
        """
        Computes the vic reg loss for the predicted features using the pseudo sam label masks

        Inputs:
            pred_key: [B, Z, H, W] tensor of predicted single scan vs multi scan features
            pred_mv_key: [B, Z, H, W] tensor of predicted multi view features
            lab_key: [B, 1, Z, H, W] tensor of pseudo sam label masks
        """
        pred = tensor_dict[self.pred_key]
        pred_mv = tensor_dict[self.pred_mv_key]
        # [B, 1, Z, H, W] -> [B, Z, H, W]
        gt = tensor_dict[self.lab_key].squeeze()
        fov_mask = tensor_dict[self.fov_key]
        B, Z, H, W = pred.shape

        if self.lab_key == "inputs/3d_ssc_label":
            with torch.no_grad():
                gt = gt / (torch.sum(gt, dim=1, keepdim=True) + 1e-5)
                gt = torch.argmax(gt, dim=1)  # [B, C, H, W] -> [B, H, W]
        else:
            # Mask labels are not consistent, remap
            gt = utils.remap_labels_in_batch(gt, ignore_idx=0)

        # TODO debug gt here
        if DEBUG_VICREG_LOSS:
            import cv2
            features = torch.stack([gt, gt], axis=0)
            fsc_image = visualize_bev_label(SAM_LABEL_DIR, features)
            cv2.imwrite("fsc_image.png", fsc_image)

        # 1 Compute fov and valid label masks
        mask = fov_mask & (gt != 0)

        # 2 Compute invariance between pairwise patches
        preds = torch.cat([pred.unsqueeze(1), pred_mv.unsqueeze(1)], dim=1)
        sim_loss = self.compute_pairwise_loss(preds, gt, mask)  # Batchwise

        # 3 Compute variance across patches with dissimilar labels
        # [B, Z, H, W] -> [B, H, W, Z] -> [B*H*W, Z]
        z1 = pred.permute(0, 2, 3, 1).reshape(B*H*W, Z)
        # [B, Z, H, W] -> [B, H, W, Z] -> [B*H*W, Z]
        z2 = pred_mv.permute(0, 2, 3, 1).reshape(B*H*W, Z)

        gt_flat = gt.reshape(B*H*W,)
        mask_flat = mask.reshape(B*H*W,)

        z1 = z1[mask_flat].view(-1, Z)
        z2 = z2[mask_flat].view(-1, Z)

        if self.use_grid_sample:
            # Use grid sample to sample features from the predicted features

            # TODO Fix grid sample to mask out using gt
            z1_samp, z1_labels = tu.grid_sample(pred, gt)
            z2_samp, z2_labels = tu.grid_sample(pred_mv, gt)

            std_loss = self.compute_variance_loss(z1_samp, z1_labels) + \
                self.compute_variance_loss(z2_samp, z2_labels)
        else:
            gt_flat = gt_flat[mask_flat].view(-1,)
            std_loss = self.compute_variance_loss(z1, gt_flat) + \
                self.compute_variance_loss(z2, gt_flat)

        # 4 Compute covariance loss between all patches from all views
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) +\
            self.off_diagonal(cov_y).pow_(2).sum().div_(z2.shape[1])
        import pdb
        pdb.set_trace()
        loss = self.sim_coeff * sim_loss + self.std_coeff * \
            std_loss + self.cov_coeff * cov_loss

        return {
            f'{self.task}/vicreg_loss': loss,
            f'{self.task}/vicreg_sim_loss': self.sim_coeff * sim_loss,
            f'{self.task}/vicreg_std_loss': self.std_coeff * std_loss,
            f'{self.task}/vicreg_cov_loss': self.cov_coeff * cov_loss
        }, {}

class MaxEntIRLLoss(Loss):
    def __init__(self, config):
        super(MaxEntIRLLoss, self).__init__(config.name, config)

        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.fov_key = config['fov_key']
        self.map_ds = config.get('map_ds', 2)
        self.map_sz = config.get('map_sz', [64, 128])
        self.bandwidth = config.get('bandwidth', 0.2)
        self.kernel = config.get('kernel', 'epanechnikov')
        self.maxent_weight = config.get('maxent_weight', 1.0)
        self.reward_weight = config.get('reward_weight', 0.1)
        self.use_fov_mask = config.get('use_fov_mask', False)

        # Modification for counterfactuals
        self.alpha = config.get('alpha', None)
        self.cf_key = config.get('cf_key', None)

        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    # @staticmethod
    # def compute_expert_visitation(gt, map_ds, map_sz):
    #     """
    #     Compute mean expert feature visitation frequencies
    #     Inputs:
    #         gt - [B, T, 3, 3] tensor of expert SE(2) poses in BEV
    #     Outputs:
    #         xy - [B, T, 2] tensor of visitation indices
    #         visit_counts - [B, H, W] tensor of visitation counts (normalized from to [0, 1])
    #     """
    #     if gt.ndim == 3:
    #         B, T, _ = gt.shape
    #         xy = gt
    #     else:
    #         B, T, _, _ = gt.shape
    #         xy = gt[:, :, :2, 2]

    #     # Preprocess export poses to visitation indices
    #     xy = xy.long() // map_ds  # [B, T, 2]
    #     x_coords = xy[:, :, 0].clamp(0, map_sz[0]-1)
    #     y_coords = xy[:, :, 1].clamp(0, map_sz[1]-1)
    #     xy_indices = x_coords * map_sz[1] + y_coords  # [B, T]

    #     H, W = map_sz

    #     # Compute visitation counts, removing duplicate states
    #     visit_counts = torch.zeros(B, H*W, dtype=torch.long, device=gt.device)
    #     visit_counts[torch.arange(B).unsqueeze(1).expand(-1, T).flatten(), xy_indices.flatten()] = 1
    #     visit_counts = visit_counts.view(B, H, W)
    #     # visit_counts = torch.zeros(B, H*W, dtype=torch.long, device=gt.device)
    #     # visit_counts = torch.scatter_add(
    #     #     visit_counts, 1, xy_indices, torch.ones_like(xy_indices))
    #     # visit_counts = visit_counts.view(B, H, W)
    #     import pdb; pdb.set_trace()
    #     return xy, visit_counts.float()  # Keep in same scale as predicted visitation

    @staticmethod
    def compute_expert_visitation(gt, map_ds, map_sz):
        """
        Compute mean expert feature visitation frequencies
        Inputs:
            gt - [B, T, 3, 3] tensor of expert SE(2) poses in BEV
        Outputs:
            interpolated_points - [B, N, 2] tensor of all interpolated visitation coordinates (xy)
            visit_counts - [B, H, W] tensor of visitation counts (normalized from 0 to 1)
        """
        if gt.ndim == 3:
            B, T, _ = gt.shape
            xy = gt
        else:
            B, T, _, _ = gt.shape
            xy = gt[:, :, :2, 2]

        # Preprocess expert poses to visitation indices
        xy = xy / map_ds  # Normalize by map downsample factor
        H, W = map_sz

        # Initialize visitation counts
        visit_counts = torch.zeros(B, H, W, dtype=torch.float32, device=gt.device)

        # Compute line segments between consecutive points
        start_points = xy[:, :-1]  # [B, T-1, 2]
        end_points = xy[:, 1:]  # [B, T-1, 2]

        # Compute interpolation steps
        distances = torch.norm(end_points - start_points, dim=-1)  # [B, T-1]
        max_steps = torch.ceil(distances).long().max().item()  # Determine max interpolation steps

        # Create interpolation factors [0, 1, ..., 1] for max_steps
        t_factors = torch.linspace(0, 1, max_steps, device=gt.device).view(1, 1, -1)  # [1, 1, max_steps]

        # Interpolate points along line segments
        interpolated_points = (
            start_points.unsqueeze(2) + t_factors * (end_points - start_points).unsqueeze(2)
        )  # [B, T-1, max_steps, 2]

        # Reshape interpolated points into a single list
        interpolated_points = interpolated_points.view(B, -1, 2)  # [B, (T-1)*max_steps, 2]
        x_coords = interpolated_points[:, :, 0].clamp(0, H - 1).long()
        y_coords = interpolated_points[:, :, 1].clamp(0, W - 1).long()

        # Flatten coordinates into a single linear index
        linear_indices = x_coords * W + y_coords  # [B, (T-1)*max_steps]

        # Accumulate visitation counts
        flat_visit_counts = visit_counts.view(B, -1)  # [B, H*W]
        flat_visit_counts.scatter_add_(
            1, linear_indices, torch.ones_like(linear_indices, dtype=torch.float32)
        )

        # Remove keep positions that are visited multiple times
        flat_visit_counts[flat_visit_counts > 1] = 1

        # Reshape back to original shape
        visit_counts = flat_visit_counts.view(B, H, W)

        return interpolated_points, visit_counts

    @staticmethod
    def compute_expert_visitation(gt, map_ds, map_sz):
        """
        Compute mean expert feature visitation frequencies
        Inputs:
            gt - [B, T, 3, 3] tensor of expert SE(2) poses in BEV
        Outputs:
            interpolated_points - [B, T, 2] tensor of all visitation coordinates (xy)
            visit_counts - [B, H, W] tensor of visitation counts (normalized from 0 to 1)
        """
        if gt.ndim == 3:
            B, T, _ = gt.shape
            xy = gt
        else:
            B, T, _, _ = gt.shape
            xy = gt[:, :, :2, 2]

        # Preprocess expert poses to visitation indices
        xy = xy / map_ds  # Normalize by map downsample factor
        H, W = map_sz

        # Initialize visitation counts
        visit_counts = torch.zeros(B, H, W, dtype=torch.float32, device=gt.device)

        # Compute line segments between consecutive points
        start_points = xy[:, :-1]  # [B, T-1, 2]
        end_points = xy[:, 1:]  # [B, T-1, 2]

        # Compute interpolation steps
        distances = torch.norm(end_points - start_points, dim=-1)  # [B, T-1]
        max_steps = torch.ceil(distances).long().max().item()  # Determine max interpolation steps

        # Create interpolation factors [0, 1, ..., 1] for max_steps
        # t_factors = torch.linspace(0, 1, max_steps, device=gt.device).view(1, 1, -1)  # [1, 1, max_steps]
        t_factors = torch.linspace(0, 1, max_steps, device=gt.device).view(1, 1, -1, 1)  # [1, 1, max_steps, 1]

        # Interpolate points along line segments
        interpolated_points = (
            start_points.unsqueeze(2) + t_factors * (end_points - start_points).unsqueeze(2)
        )  # [B, T-1, max_steps, 2]

        # Reshape interpolated points into a single list
        interpolated_points = interpolated_points.view(B, -1, 2)  # [B, (T-1)*max_steps, 2]

        # Append the last point for each trajectory to preserve T horizon
        last_points = xy[:, -1:]  # [B, 1, 2]
        interpolated_points = torch.cat([interpolated_points, last_points], dim=1)  # [B, N+1, 2]

        # Clamp to valid grid indices
        x_coords = interpolated_points[:, :, 0].clamp(0, H - 1).long()
        y_coords = interpolated_points[:, :, 1].clamp(0, W - 1).long()

        # Flatten coordinates into a single linear index
        linear_indices = x_coords * W + y_coords  # [B, N+1]

        # Accumulate visitation counts
        flat_visit_counts = visit_counts.view(B, -1)  # [B, H*W]
        flat_visit_counts.scatter_add_(
            1, linear_indices, torch.ones_like(linear_indices, dtype=torch.float32)
        )

        # Reshape back to original shape
        visit_counts = flat_visit_counts.view(B, H, W)

        # Remove keep positions that are visited multiple times
        visit_counts[visit_counts > 1] = 1 

        return interpolated_points, visit_counts

    def loss(self, tensor_dict):
        """
        Inputs:
            pred_key: [B, V, 2] tensor of predicted trajectories
            lab_key: [B, V, 2] tensor of expert trajectories
        """
        exp_svf = tensor_dict[self.pred_key]
        gt = tensor_dict[self.lab_key]
        fov_mask = tensor_dict[self.fov_key]
        reward_preds = tensor_dict['outputs/traversability_preds']
        state_features = tensor_dict['outputs/input_view']
        reward_preds = reward_preds.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        _, Ho, Wo = fov_mask.shape
        _, H, W = exp_svf.shape
        ds = Wo // W
        fov_mask = tu.resize_and_crop(fov_mask.unsqueeze(
            1).byte(), (Ho//2, Wo//2), (0, H, 0, W))
        fov_mask = fov_mask.squeeze(1).bool()

        # Compute difference in expert and predicted visitation frequencies
        xy, svf = self.compute_expert_visitation(gt, self.map_ds, self.map_sz)

        # Mask and normalize visitation frequencies
        if self.use_fov_mask:
            svf = svf * fov_mask.float()
            exp_svf = exp_svf * fov_mask.float()
        svf = svf / (svf.sum(dim=(1, 2), keepdim=True) + 1e-5)
        exp_svf = exp_svf / (exp_svf.sum(dim=(1, 2), keepdim=True) + 1e-5)

        # import cv2
        # cv2.imwrite("test.png", cv2.normalize(svf[0].detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX))
        cf_svf_total = torch.zeros_like(svf)
        exp_svf_total = exp_svf.clone()
        if self.cf_key is not None and self.alpha is not None:
            cf_dict_list = tensor_dict[self.cf_key]

            for idx, cf_dict in enumerate(cf_dict_list):
                # Convert cf to state visitation frequencies
                if cf_dict is None:
                    continue
                cf_traj = cf_dict['trajectories']
                invalid_traj = cf_traj[cf_dict['rank']>0] # [N, T, 2]
                if invalid_traj.shape[0] == 0:
                    continue

                # Keep visitation distribution to # of times visited per trajectory
                invalid_traj = torch.from_numpy(invalid_traj).to(svf.device)
                cf_xy, cf_svf = self.compute_expert_visitation(
                    invalid_traj, self.map_ds, self.map_sz)

                cf_svf = cf_svf.sum(dim=0) # [H, W]
                cf_svf = cf_svf / (cf_svf.sum(dim=(0, 1), keepdim=True) + 1e-5)

                # Debug visualization
                if DEBUG_MAXENT_LOSS:
                    with torch.no_grad():
                        import cv2
                        import numpy as np
                        visit_counts_img = cf_svf.detach().cpu().numpy()
                        visit_counts_img = cv2.normalize(
                            visit_counts_img, None, 0, 255, cv2.NORM_MINMAX)
                        cv2.imwrite(f"test.png", visit_counts_img)

                # Mix policy and counterfactual visitation frequencies
                exp_svf[idx] = self.alpha * cf_svf + (1 - self.alpha) * exp_svf[idx]

                # Accumulate cf_svf for validation
                cf_svf_total[idx] = cf_svf
        assert torch.all(exp_svf >= 0), "Negative expert visitation frequencies"
        assert torch.all(svf >= 0), "Negative predicted visitation frequencies"

        if self.use_fov_mask:
            # Mask out visitation frequencies outside of the FOV
            ones_mask = torch.ones_like(reward_preds, dtype=torch.float32)
            ones_mask[~fov_mask] = 0
            reward_preds = reward_preds * ones_mask # Differentiable masking
        
        # Compute sum of rewards for expert and predicted visitation frequencies
        svf_rewards = (svf * reward_preds).sum(dim=(1, 2))
        exp_svf_rewards = (exp_svf * reward_preds).sum(dim=(1, 2))

        # Average the difference in summed rewards across batch
        mean_exp_svf_rewards = torch.mean(exp_svf_rewards)
        mean_svf_rewards = torch.mean(svf_rewards)
        visitation_loss = mean_exp_svf_rewards - mean_svf_rewards
        
        # Regularize gradient of reward w.r.t states SMODICE 
        # (https://github.com/JasonMa2016/SMODICE/blob/d860c353c9abecd1b7eb92e2166f825f83ecb901/discriminator_pytorch.py#L133)
        reward_penalty = torch.tensor(0.0, device=svf.device)
        if reward_preds.requires_grad and self.reward_weight > 0:
            reward_grad = torch.autograd.grad(
                outputs=reward_preds.sum(),
                inputs=state_features,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            reward_grad_norm = reward_grad.norm(2, dim=1)
            reward_penalty = ((reward_grad_norm - 1)**2).mean()

        loss = self.maxent_weight * visitation_loss + self.reward_weight * reward_penalty

        with torch.no_grad():
            # Compute sum of rewards along suboptimal and expert trajectories
            cf_rewards = (cf_svf_total * reward_preds).sum(dim=(1, 2))
            opt_rewards = (exp_svf_total * reward_preds).sum(dim=(1, 2))

            # Mask out batch indices with no counterfactuals
            valid_indices = cf_rewards != 0

            # Compute sum of rewards for expert and predicted visitation frequencies
            cf_rewards = cf_rewards[valid_indices].sum()
            opt_rewards = opt_rewards[valid_indices].sum()

        if DEBUG_MAXENT_LOSS:
            with torch.no_grad():
                # print(f"MaxEntIRL Loss: {loss}")
                import cv2
                import numpy as np
                # test_fov_mask = fov_mask[0].detach().cpu().numpy()
                # test_fov_mask = test_fov_mask.astype(np.uint8)
                # cv2.imwrite("fov_mask.png", test_fov_mask * 255)

                visit_counts_img = svf[0].detach().cpu().numpy()
                visit_counts_img = cv2.normalize(
                    visit_counts_img, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite("maxentgt.png", visit_counts_img)

                visit_counts_img = exp_svf[0].detach().cpu().numpy()
                visit_counts_img = cv2.normalize(
                    visit_counts_img, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite("maxentpred.png", visit_counts_img)

        meta_dict = {
            "reward_penalty": self.reward_weight * reward_penalty,
            "mean_expected_svf_rewards": mean_exp_svf_rewards,
            "mean_svf_rewards": mean_svf_rewards,
            "sum_cf_rewards": cf_rewards,
            "sum_opt_rewards": opt_rewards
        }
        return {f'maxentirl_loss': loss}, meta_dict

class BCActionLoss(Loss):
    def __init__(self, config):
        super(BCActionLoss, self).__init__(config.name, config)

        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.fov_key = config['fov_key']

        self.register_buffer('actions', torch.tensor([
            [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]
        ]).float())

        self.bce_loss = torch.nn.BCELoss(reduction='mean')

    def loss(self, tensor_dict):
        """
        Inputs:
            pred_key: [B, T, 8] tensor of predicted actions
            lab_key: [B, T, 3, 3] tensor of expert SE2 poses
        """
        pred = tensor_dict[self.pred_key]
        gt = tensor_dict[self.lab_key]
        fov_mask = tensor_dict[self.fov_key]
        B, T, _ = pred.shape

        # Compute loss for each time step
        loss = 0
        for t in range(1, T):
            prev_gt = gt[:, t-1, :2, 2]
            curr_gt = gt[:, t, :2, 2]
            # Execute each action on the prev_gy adn determine which one is closest to curr_gt
            action_deltas = self.actions - (curr_gt - prev_gt).unsqueeze(1)
            action_deltas = torch.norm(action_deltas, dim=-1)
            closest_action = torch.argmin(action_deltas, dim=1)
            closest_action = F.one_hot(closest_action, num_classes=8).float()

            # Compute loss for each time step
            loss += self.bce_loss(pred[:, t], closest_action)
        loss /= T  # Average over time steps

        return {f'bc_action_loss': loss}, {}

class TREXLoss(Loss):
    def __init__(self, config):
        super(TREXLoss, self).__init__(config.name, config)

        self.pred_key = config['pred_key']
        self.lab_key = config['lab_key']
        self.fov_key = config['fov_key']

        self.map_ds = config.get('map_ds', 2)
        self.map_sz = config.get('map_sz', [64, 128])

        self.l1_reg = config.get('l1_reg', 0.1)
        self.bce_loss = torch.nn.BCELoss(reduction='sum')

    def preprocess_counterfactuals(self, cf, device):
        """Preprocesses cf for loss computation"""
        # 1 Load counterfactuals (Already in BEV frame) 
        poses = torch.from_numpy(
            np.array([cf_dict['trajectories'] for cf_dict in cf])
        ).to(device) # [B, N, T, 2]
        ranks = torch.from_numpy(
            np.array([cf_dict['rank'] for cf_dict in cf])
        ).to(device) # [B, N] 0 for highest ranked, 1 for second highest, etc.

        # Downsample pose to grid map resolution
        poses = (poses / self.map_ds).round().long()
        poses = poses.clamp(torch.tensor([0, 0]).to(device), torch.tensor(self.map_sz).to(device)-1)

        return poses, ranks

    def loss(self, tensor_dict):
        """
        Inputs:
            pred_key: [B, T, 8] tensor of predicted actions
            lab_key: [B, N, T, 2] list of counterfactual expert xy positions
        """
        pred = tensor_dict[self.pred_key]
        cf = tensor_dict[self.lab_key]
        device = pred.device

        with torch.no_grad():
            # 1 Load counterfactuals (Already in BEV frame) 
            poses, ranks = self.preprocess_counterfactuals(cf, device)

        # 2 Compute cross entropy style preference loss (TREX)
        l1_loss = torch.mean(torch.abs(pred))
        B, _, H, W = pred.shape
        total_loss = 0
        num_pairs = 0
        for b in range(B):
            # Get the prediction for the current batch element
            pred_b = pred[b].squeeze()  # [1, H, W]

            # Preferred and non-preferred mask for the current batch
            pref_mask = ranks[b] == 0  # [N]
            not_pref_mask = ranks[b] > 0  # [N]

            # Select preferred and non-preferred poses for the current batch
            pref_poses = poses[b, pref_mask]  # [N_pref, T, 2]
            not_pref_poses = poses[b, not_pref_mask]  # [N_not_pref, T, 2]

            # Vectorized indexing to compute reward sum for preferred and non-preferred trajectories
            reward_pref_sum = pred_b[pref_poses[:, :, 0], pref_poses[:, :, 1]].sum(dim=1)  # [N_pref]
            reward_not_pref_sum = pred_b[not_pref_poses[:, :, 0], not_pref_poses[:, :, 1]].sum(dim=1)  # [N_not_pref]

            # Compute cross entropy style preference loss (TREX) for the current batch element
            N_pref = reward_pref_sum.shape[0]
            N_not_pref = reward_not_pref_sum.shape[0]
            # # reward_pairs = torch.cartesian_prod(reward_pref_sum, reward_not_pref_sum)
            reward_pairs = torch.stack([
                reward_pref_sum.repeat(N_not_pref),
                reward_not_pref_sum.repeat(N_pref)
            ], dim=1)
            labels = torch.ones(N_pref * N_not_pref, device=device)

            # # Randomly shuffle the reward pairs
            # rand_indices = torch.randperm(reward_pairs.shape[0])
            # reward_pairs = reward_pairs[rand_indices]
            # labels = labels[rand_indices]

            # Apply log-sum-exp trick for numerical stability
            normalizing_constant = torch.logsumexp(reward_pairs, dim=1)
            # reward_pairs[:, 0] = reward_pairs[:, 0] - normalizing_constant
            # reward_pairs[:, 1] = reward_pairs[:, 1] - normalizing_constant
            reward_pairs = torch.stack([
                reward_pairs[:, 0] - normalizing_constant,
                reward_pairs[:, 1] - normalizing_constant
            ], dim=1)

            # Compute cross entropy in terms of software ylog(p) + (1-y)log(1-p)
            p1 = reward_pairs[:, 0] / (reward_pairs[:, 0] + reward_pairs[:, 1] + 1e-6)
            p1 = F.softmax(p1, dim=0)
            cls_loss = self.bce_loss(p1, labels)
            total_loss += cls_loss

            # total_loss += -(labels * torch.log(p1) + (1-labels) * torch.log(1-p1))
            num_pairs += reward_pairs.shape[0] 
            # import pdb; pdb.set_trace()
            # # Compute cross entropy loss
            # total_loss += -torch.log(torch.exp(reward_pairs[:,0]) / (torch.exp(reward_pairs[:,0]) + torch.exp(reward_pairs[:,1]))).sum()
        total_loss /= num_pairs + self.l1_reg*l1_loss
        return {f'trex_loss': total_loss}, {}
        


