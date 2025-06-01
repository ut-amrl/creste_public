import torch.nn.functional as F
import torch_scatter
from creste.utils.utils import warp, make_labels_contiguous_vectorized
import math
import torch
from torch import nn
# from torchvision.transforms import v2
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

import kornia
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module='kornia.feature.lightglue')
import kornia.geometry.transform as ktf

DEBUG_AUGMENTATION = 0


class ImageAugmentation(object):
    def __init__(
        self, 
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        gamma=None,
        gamma_p=0.0
    ):
        """
        Initialize the transform with the desired parameters.

        Parameters:
        - brightness (float or tuple): How much to jitter brightness.
        - contrast (float or tuple): How much to jitter contrast.
        - saturation (float or tuple): How much to jitter saturation.
        - hue (float or tuple): How much to jitter hue.
        - gamma (float or tuple, optional): Range for random gamma correction.
        - gamma_p (float): Probability of applying gamma. 0.0 means disabled.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.gamma_p = gamma_p
        
        # Build the pipeline once
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """Helper to construct the augmentation pipeline."""
        transforms_list = []
        
        # Always apply ColorJitter if any of its parameters are nonzero
        if any([self.brightness, self.contrast, self.saturation, self.hue]):
            transforms_list.append(
                kornia.augmentation.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                    keepdim=True
                )
            )
        
        # Apply RandomGamma only if gamma is specified and gamma_p > 0
        if self.gamma is not None and self.gamma_p > 0:
            transforms_list.append(
                kornia.augmentation.RandomGamma(
                    gamma=self.gamma,
                    p=self.gamma_p,
                    keepdim=True
                )
            )
        
        # If no transforms are added, just use an identity (no-op)
        if not transforms_list:
            return nn.Identity()
        
        return nn.Sequential(*transforms_list)

    def renew_augmentation(self):
        """
        Sample new augmentation parameters by rebuilding the pipeline.
        """
        self.pipeline = self._build_pipeline()

    def __call__(self, img, keep_aug=False):
        """
        Apply the transforms to the given image.

        Parameters:
        - img (torch.Tensor): The image to transform.
        - keep_aug (bool): If True, keep the same sampled parameters for 
                           the augmentation steps as previously used.
                           Otherwise, re-sample them.

        Returns:
        - torch.Tensor: The transformed image.
        """
        if keep_aug:
            # Use previously sampled parameters
            return self.pipeline(img, params=self.pipeline._params)
        else:
            # Resample parameters
            return self.pipeline(img)

class DepthAugmentation(object):
    def __init__(self, dropout_prob=0.1, calib_error_mean=[0.0, 0.0, 0.0],
        calib_error_std=[0.02, 0.02, 0.01], depth_noise_std=0.2):
        """
        Initializes the LiDAR augmentation parameters.

        Args:
            dropout_prob (float): Probability of LiDAR point dropout.
            miscalibration_translation (tuple): Translation offsets (tx, ty) for miscalibration in normalized space.
            miscalibration_rotation (float): Rotation angle for miscalibration in radians.
            depth_noise_std (float): Standard deviation of Gaussian noise added to depth.
        """
        self.dropout_prob = dropout_prob
        self.calib_error_mean = calib_error_mean
        self.calib_error_std = calib_error_std
        self.depth_noise_std = depth_noise_std

    def simulate_lidar_dropout(self, depth_map):
        """Simulates LiDAR dropout by randomly masking depth values."""
        mask = torch.rand_like(depth_map) > self.dropout_prob
        return depth_map * mask

    def simulate_miscalibration(self, depth_map):
        """Simulates camera-LiDAR miscalibration with translation and rotation sampled from a normal distribution."""
        _, H, W = depth_map.shape  # 1xHxW

        # Sample translation (tx, ty) and rotation (angle in degrees) from normal distributions
        noise = torch.normal(mean=torch.tensor(self.calib_error_mean), std=torch.tensor(self.calib_error_std))

        tx, ty = noise[0], noise[1]  # Translation components
        angle = noise[2] * (180.0 / torch.pi)  # Convert radians to degrees

        # Center point of the image
        center = torch.tensor([[W / 2, H / 2]])

        # Create the affine transformation matrix
        transform = ktf.get_affine_matrix2d(
            translations=torch.tensor([[tx, ty]]),  # Translation in pixels
            center=center,
            scale=torch.tensor([[1.0, 1.0]]),  # No scaling
            angle=torch.tensor([angle])  # Angle in degrees
        )
        # Convert 3x3 affine matrix to 2x3 for warp_affine
        transform = transform[:, :2, :]  # Drop the last row

        # Apply the transformation to the depth map
        depth_map_transformed = ktf.warp_affine(depth_map.unsqueeze(0), transform, dsize=(H, W)).squeeze(0)
        return depth_map_transformed

    def simulate_depth_noise(self, depth_map):
        """Adds Gaussian noise to depth values."""
        noise = torch.randn_like(depth_map) * self.depth_noise_std
        return depth_map + noise

    def __call__(self, depth_map):
        """
        Applies all augmentations to the input depth map.

        Args:
            depth_map (torch.Tensor): Input depth map of shape (1, H, W).

        Returns:
            torch.Tensor: Augmented depth map.
        """
        # Ensure the input is of the correct shape
        assert depth_map.ndim == 3 and depth_map.shape[0] == 1, "Input depth map must have shape (1, H, W)."

        depth_map = self.simulate_lidar_dropout(depth_map)
        depth_map = self.simulate_miscalibration(depth_map)
        depth_map = self.simulate_depth_noise(depth_map)
        return depth_map

class RotateAndTranslate(object):
    def __init__(self, augmentations, map_size, voxel_size):
        """
        Initialize the transform with the desired parameters.

        Parameters:
        - augmentations (list): A list of augmentations to apply.
        - map_size (list): The size of the map in meters [x, y]
        - voxel_size (float): The size of each voxel in meters [x, y]
        """
        self.augmentations = {}

        for aug in augmentations:
            kwargs = aug.copy()
            aug_name = kwargs.pop('name')

            if aug_name == 'rotate':
                max_angle = kwargs.pop('max_rotation', 0.0)
                self.augmentations['rotate'] = max_angle
            elif aug_name == 'translate':
                max_translation = kwargs.pop('max_translation', 0.0)
                self.augmentations['translate'] = max_translation
            else:
                raise ValueError(f'Augmentation {aug_name} not supported')

        self.map_size = torch.tensor(map_size).float()
        self.voxel_size = torch.tensor(voxel_size).float()
        self.center = (self.map_size / self.voxel_size /
                       2).float().unsqueeze(0)
        self.scale = torch.tensor([1.0, 1.0]).float().unsqueeze(0)

        assert (len(self.augmentations) > 0)

    def transform_map(self, map, R_init=None, interpolation='nearest'):
        """
        Assumes a relatively constant height map (i.e. no vertical translation)
        Warps the given map according to the SE2 transformation matrix

        Inputs:
            map: [H, W, C] tensor
        Outputs:
            tmap: [H, W, C] tensor
            mask: [H, W] tensor
        """
        map = map.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        RT = self.mapRT
        if R_init is not None:
            RT = RT @ R_init

        tmap, mask = warp(map, RT, interpolation=interpolation)
        tmap = tmap.squeeze(0).permute(1, 2, 0)
        mask = mask.squeeze(0)

        if DEBUG_AUGMENTATION:
            print("Debugging bev label augmentation")

            if map.shape[1] > 1:
                # Sanity check transformation
                cls_map = torch.argmax(map, dim=1)
                cls_tmap = torch.argmax(tmap, dim=-1).unsqueeze(0)
                from ssc_utils.visualization import visualize_bev_label
                visualize_bev_label(cls_map, cls_tmap, "test.png")
            else:
                import cv2
                import numpy as np
                # Show the original map and tmap side by side in opencv image
                cls_map = map.squeeze()
                cls_map = cls_map.detach().cpu().numpy()
                cls_tmap = tmap.squeeze(-1)
                cls_tmap = cls_tmap.detach().cpu().numpy()
                out_img = np.concatenate(
                    (cls_map, cls_tmap), axis=1).astype(np.uint8)

                out_img = out_img * 255

                cv2.imwrite("testmask.png", out_img)

        return tmap, mask

    def transform(self, inputs):
        '''
            inputs: 4xN tensor homogeneous xyz tensor
        '''
        assert inputs.shape[0] == 4, "Points must be Nx4 tensor"

        # transform xyz
        inputs = self.RT @ inputs

        return inputs

    def renew_transformation(self):
        """
        Sample a SE3 transformation matrix for BEV map augmentation
        """
        RT = torch.eye(4)

        if 'translate' in self.augmentations:
            max_translation = self.augmentations['translate']
            RT[:2, 3] = (2*torch.rand(2)-1) * max_translation

        if 'rotate' in self.augmentations:
            max_angle = self.augmentations['rotate']
            angle = (2*torch.rand(1)[0]-1) * max_angle
            angle_rad = angle * torch.pi / 180

            RT[:2, :2] = torch.tensor([
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)]
            ])

        # Computer mapRT
        angle = angle.reshape(1)
        M = kornia.geometry.transform.get_rotation_matrix2d(
            self.center, angle, self.scale)

        self.RT = RT
        self.mapRT = M

    def compute_transformation_fromSE3(self, RT):
        """
        Compute the map transformation matrix from a SE3 transformation matrix
        """
        # Compute mapRT
        R = RT[:2, :2]
        t = RT[:2, 3]

        # Convert t from munits of meters to pixels and then center it
        offset = (t / self.voxel_size).float().unsqueeze(0)

        # Compute angle
        angle = torch.atan2(R[1, 0], R[0, 0]) * 180 / torch.pi  # degrees
        angle = angle.reshape(1)

        M = kornia.geometry.transform.get_affine_matrix2d(
            offset, self.center, self.scale, angle
        )

        return M


def extract_max_per_class(tensor, max_per_class=100, return_indices=True):
    # Find unique classes in the tensor
    unique_classes = torch.unique(tensor)

    # Initialize an empty tensor for the result
    result_indices = torch.LongTensor().to(tensor.device)

    # Iterate over each class to select up to max_per_class elements
    for class_index in unique_classes:
        # Find indices of all occurrences of the current class
        indices = (tensor == class_index).nonzero(as_tuple=False).reshape(-1)

        # If more than max_per_class elements, randomly select max_per_class of them
        if indices.size(0) > max_per_class:
            # Generate a random permutation of indices and select the first max_per_class
            selected_indices = indices[torch.randperm(indices.size(0))[
                :max_per_class]]
        else:
            # If fewer elements than max_per_class, select all of them
            selected_indices = indices

        # Concatenate the selected indices for this class to the result indices
        result_indices = torch.cat((result_indices, selected_indices))

    # Now, result_indices contains the indices of the selected elements
    # You can return the selected elements or the indices
    if return_indices:
        return result_indices
    return tensor[result_indices]


def compute_pixel_bevoverlap_loss(
    bev_coords, pred, gt, img=None
):
    """
    Construct a mask that indicates the pixels that have overlapping bev coords. The first index for
    bev_coords is the anchor view with the rest being the other views.

    Inputs:
    H (int): Height of each image
    W (int): Width of each image
    bev_coords - [B, Vp1, H, W, 2] tensor of BEV coordinates
    feats - [B, Vp1, C, H, W] tensor of image features

    Outputs:
    correspondences - [B, H*W, H*W*V] tensor of pixel correspondences
    """
    B, Vp1, H, W, _ = bev_coords.shape
    V = Vp1 - 1

    pred = pred.view(B, Vp1, H, W, -1)
    gt = gt.view(B, Vp1, H, W, -1)
    pred_anchor_feats = pred[:, 0, :, :, :]  # [B,  H, W, C]
    gt_anchor_feats = gt[:, 0, :, :, :]  # [B, H, W, C]
    pred_aug_feats = pred[:, 1:, :, :, :]
    gt_aug_feats = gt[:, 1:, :, :, :]

    bev_coords = bev_coords.view(B, Vp1, H*W, 2)
    anchor_view = bev_coords[:, 0, :, :].view(
        B, H, W, 2).unsqueeze(1)  # [B, 1, H, W, 2]
    aug_views = bev_coords[:, 1:, :, :].view(B, V, H, W, 2)  # [B, V, H, W, 2]

    # Compute the bev coord distance between the anchor view and the other views to construct overlap mask
    loss = 0
    if Vp1 > 1:
        for b in range(B):
            cur_anchor_view = anchor_view[b].view(H*W, 1, 2)
            cur_aug_views = aug_views[b].permute(1, 2, 0, 3).reshape(H*W, V, 2)

            cur_aug_views_flat = cur_aug_views.reshape(1, H*W*V, 2)
            cur_anchor_view_flat = cur_anchor_view.reshape(1, H*W, 2)

            patch_dist = torch.cdist(
                cur_anchor_view_flat, cur_aug_views_flat, p=2)

            # Valid pixels have at least one view overlapped with anchor
            valid_pixels = torch.any(patch_dist < 1, dim=1).view(
                H, W, V).permute(2, 0, 1)  # [V, H, W]

            loss += F.mse_loss(
                pred_aug_feats[b][valid_pixels], gt_aug_feats[b][valid_pixels], reduction='mean'
            )

            # TODO Debug this by visualizing hte overlapping pixel for first view
            if DEBUG_AUGMENTATION:
                import cv2
                import numpy as np

                with torch.no_grad():
                    for v in range(V):
                        test_img = np.zeros((H, W, 3), dtype=np.uint8)
                        test_img[valid_pixels[v].cpu().numpy()] = [
                            255, 255, 255]

                        if img is not None:
                            anchor_img = (img[b, 0].permute(1, 2, 0)[
                                          :, :, :3].cpu().numpy() * 255).astype(np.uint8)
                            anchor_img_reduced = cv2.resize(anchor_img, (W, H))

                            # Alpha blend aug view image and mask
                            aug_image = (
                                img[b, v+1].permute(1, 2, 0)[:, :, :3].cpu().numpy() * 255).astype(np.uint8)
                            aug_image_reduced = cv2.resize(aug_image, (W, H))
                            weighted_img = cv2.addWeighted(
                                aug_image_reduced, 0.5, test_img, 0.5, 0)

                            # concatenate anchor and overlapping views
                            concat_img = np.concatenate(
                                (anchor_img_reduced, weighted_img), axis=1)

                            cv2.imwrite("testbevoverlap.png", concat_img)

    # Compute loss on all pixels in anchor view
    loss += F.mse_loss(pred_anchor_feats, gt_anchor_feats)

    return loss


def median_filter_2d(image, kernel_size=3):
    """
    Inputs:
        image - [B, C, H, W] tensor
        kernel_size - int
    Outputs:
        median_filtered - [B, C, H, W] tensor
    """
    _, _, H, W = image.shape

    # Convert image to a 4D tensor (1, 1, H, W)
    image = image.float()

    # Define the padding size
    padding = kernel_size // 2

    # Pad the image to handle borders
    image_padded = F.pad(
        image, (padding, padding, padding, padding), mode='reflect')

    # Unfold the image to get sliding windows
    unfolded = image_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

    # Reshape to (N, C, H, W, kH*kW)
    unfolded = unfolded.contiguous().view(1, 1, H, W, -1)

    # Mask out zeros by setting them to a large value and sort
    unfolded[unfolded == 0] = float('inf')
    unfolded, _ = unfolded.sort(dim=-1)

    # Find the median by excluding the inf values
    num_non_zero = (unfolded != float('inf')).sum(dim=-1)
    median_indices = (
        num_non_zero // 2).long().clamp(max=kernel_size*kernel_size - 1)

    # Gather the median values
    median_filtered = unfolded.gather(
        dim=-1, index=median_indices.unsqueeze(-1)).squeeze(-1)

    # Replace inf values back to zero
    median_filtered[median_filtered == float('inf')] = 0

    return median_filtered.long()

def expand_filter_2d(tensor, kernel_size=3):
    """
    Expands the regions around non-zero values in a BxCxHxW tensor by setting the kernel around each non-zero value.
    The values in the expanded region will be the maximum value found within that kernel window.

    Args:
        tensor (torch.Tensor): Tensor of shape (B, C, H, W) containing arbitrary values.
        kernel_size (int): Size of the kernel around each non-zero value. Default is 3 (3x3).

    Returns:
        torch.Tensor: Tensor with values expanded around the neighborhood of original non-zero values.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    # Create a convolutional kernel filled with 1s
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=tensor.device)

    # Apply the convolution to spread the influence of non-zero values
    # Use F.max_pool2d to propagate the maximum value in the neighborhood
    expanded_tensor = F.max_pool2d(tensor.float(), kernel_size=kernel_size, stride=1, padding=padding)

    return expanded_tensor


def create_trapezoidal_fov_mask(
    H, W, fov_top_angle=50, fov_bottom_angle=40, near=10, far=50
):
    """
    Create a trapezoidal field of view mask of shape HxW facing north.

    Parameters:
    H (int): Height of the mask.
    W (int): Width of the mask.
    fov_top_angle (int): The top angle of the FOV in degrees (smaller angle).
    fov_bottom_angle (int): The bottom angle of the FOV in degrees (larger angle).
    near (int): The distance from the observer to the near edge of the trapezoid.
    far (int): The distance from the observer to the far edge of the trapezoid.

    Returns:
    torch.Tensor: A boolean mask of shape HxW.
    """
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_x, center_y = W / 2, H / 2

    # Calculate the distances and angles
    distances = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    angles = torch.atan2(x - center_x, center_y - y) * 180 / torch.pi

    # Adjust angles to be from -180 to 180
    angles[angles < -180] += 360

    # Calculate the angular boundaries at each distance
    angular_spread_top = torch.full_like(distances, fov_top_angle / 2)
    angular_spread_bottom = torch.full_like(distances, fov_bottom_angle / 2)

    # Linear interpolation for angular spread based on distance
    angular_spread = torch.where(distances <= near, angular_spread_top,
                                 torch.where(distances >= far, angular_spread_bottom,
                                             angular_spread_top + (angular_spread_bottom - angular_spread_top) * ((distances - near) / (far - near))))

    # Create the mask
    mask = torch.logical_and(distances >= near, distances <= far)
    mask = torch.logical_and(mask, torch.abs(angles) <= angular_spread)

    # import pdb; pdb.set_trace()
    # import cv2
    # import numpy as np
    # cv2.imwrite("test.png", mask.detach().cpu().numpy().astype(np.uint8) * 255)

    return mask


def prefix_dict(prefix, d, seprator='/'):
    """
    Add a prefix to dictionary keys.
    """
    return {prefix + seprator + k: v for k, v in d.items()}


def merge_dict(*args):
    """
    Merge multiple dicts and optionally add a prefix to
    the keys.

    :param args: A list. Each element can be a dict or a
                  tuple (prefix, dict)
    :return: merged dictionary
    """
    ret = dict()
    for arg in args:
        if isinstance(arg, dict):
            ret.update(arg)
        else:
            prefix, d = arg
            ret.update(prefix_dict(prefix, d))
    return ret


def merge_loss_dict(
    full_dict,
    new_dict,
):
    """
    Merge two metadata dictionaries.
    """
    for k, v in new_dict.items():
        if k not in full_dict:
            full_dict[k] = v
        else:
            full_dict[k] = v  # override prior value

    return full_dict


def get_save_paths(cfg, model_type="ssc", stage="train"):
    from datetime import datetime
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    day, time = cur_time.split('_')

    run_name = ''
    model_cfg = cfg['model']

    # Suffix path
    if model_type == 'lfd_maxentirl':
        run_name += '/%s_head%s_horizon%s' \
            % (
                cfg['model']['run_name'],
                cfg['model']['traversability_head']['name'],
                cfg['dataset']['action_horizon']
            )
        model_cfg = cfg['model']['vision_backbone']
    elif model_type == 'lfd_bc':
        run_name += '/%s_in%s_out%s_lr%s' \
            % (
                cfg['model']['run_name'],
                cfg['model']['bc_head']['in_horizon'],
                cfg['model']['bc_head']['out_horizon'],
                cfg['model']['optimizer']['lr']
            )
        model_cfg = cfg['model']['backbone']
    elif model_type == "clusterlookup":
        run_name += '%s_ncls%s_ecls%s_fkey%s_lkey%s' \
            % (
                cfg.model.run_name,
                cfg.model.n_classes,
                cfg.model.extra_clusters,
                cfg.model.feat_key,
                cfg.mode.label_key
            )

    if model_type != "cluster_probe":
        run_name = '%s_BB_%s_Head_%s_lr_%f_%s_%s_v2' \
            % (
                model_cfg['run_name'],
                model_cfg['vision_backbone']['name'],
                model_cfg['depth_head']['name'],
                model_cfg['optimizer']['lr'],
                model_cfg['discretize']['mode'],
                cfg['dataset']['infill_strat']
            ) + run_name

    if stage == "train":
        root_dir = cfg['trainer']['default_root_dir']
    elif stage == "test":
        root_dir = cfg['trainer'].get('test_root_dir', 'model_outputs')
    else:
        raise ValueError(f"Invalid stage {stage}")

    ckpt_save_dir = join(
        cfg['trainer']['default_root_dir'],
        cfg['model']['project_name'],
        run_name,
        day,
        time
    )
    if not os.path.exists(ckpt_save_dir):
        print(f'Saving model checkpoints {ckpt_save_dir}')
        os.makedirs(ckpt_save_dir)

    return ckpt_save_dir


def resize_and_crop(image, new_size, crop_bounds):
    """
    Inputs:
        image: [B, C, H, W] tensor of float32 or uint8
        new_size: (int, int) tuple
        crop_bounds: (int, int, int, int) tuple
    """
    H, W = new_size
    y1, y2, x1, x2 = crop_bounds
    image = F.interpolate(image, size=(H, W), mode='nearest')
    image = image[:, :, y1:y2, x1:x2].clone()

    return image

def grid_sample(x, label, grid_dim=8, aggregation="mean", min_feats=2, ignore_idx=0):
    """
    Given a BxFxHxW tensor, divide the grid into grid_dim x grid_dim number of patches
    and aggregate the features in each patch based on the label. The label is assumed to be
    a BxHxW tensor of class labels. This creates a list of Np aggregated features where
    each feature has a label. Then filter the features such that only the features
    with at least min_feats number of occurences are retained.

    Inputs:
        x - (torch.Tensor) BxFxHxW tensor of features
        label - (torch.Tensor) BxHxW tensor of class labels
    Outputs:
        sampled_patches - (torch.Tensor) [N, Z] tensor of aggregated features
        sampled_labels - (torch.Tensor) [N] tensor of class labels
    """
    B, Z, H, W = x.shape

    # 1 Sample grid patches
    patch_size = H // grid_dim
    grid_patches = x.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)

    # 2 Aggregate feature per patch based on label
    patch_labels = label.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size)

    # 3 Sample class_per_grid number of classes per patch
    # grid_patches_flat = grid_patches.view(B, F, grid_dim, grid_dim, -1)
    sampled_patches = torch.empty((0, Z), device=x.device, dtype=x.dtype)
    sampled_labels = torch.empty((0), device=label.device, dtype=label.dtype)
    # torch.use_deterministic_algorithms(False)
    for b in range(B):
        patch = grid_patches[b]         # [Z, grid_dim, grid_dim, Ps, Ps]
        patch_label = patch_labels[b]   # [grid_dim, grid_dim, Ps, Ps]

        Ps = patch.shape[-1]
        assert patch.shape[1] == grid_dim and patch.shape[2] == grid_dim, "Invalid patch shape"

        patch = patch.reshape(Z, grid_dim, grid_dim, Ps*Ps)
        patch_label = patch_label.reshape(grid_dim, grid_dim, Ps*Ps)

        # [grid_dim, grid_dim, #classes]
        unique_classes = torch.unique(patch_label)

        # Make patch_labels contiguous for scatter op
        patch_label_cont = make_labels_contiguous_vectorized(patch_label)

        # Compute aggregated features per class
        patch_aggr = torch_scatter.scatter(
            patch, patch_label_cont.unsqueeze(0), dim=-1, reduce=aggregation
        )  # [Z, grid_dim, grid_dim, #classes]

        # Sample class_per_grid number of classes from each patch according to frequency
        # [grid_dim, grid_dim, #classes, Z]
        patch_aggr = patch_aggr.permute(1, 2, 3, 0)
        # [grid_dim, grid_dim, #classes]
        patch_label_aggr = torch.tile(unique_classes, (grid_dim, grid_dim, 1))

        # Filter out invalid patches
        valid_mask = (patch_label_aggr != ignore_idx) & (
            torch.any(patch_aggr != 0, dim=-1))
        patch_aggr = patch_aggr[valid_mask]
        patch_label_aggr = patch_label_aggr[valid_mask]

        # Remove features with only one occurence among patches
        uq_aggr, counts_aggr = torch.unique(
            patch_label_aggr, return_counts=True)
        invalid_labels = uq_aggr[counts_aggr < min_feats][:, None]

        patch_label_mask = ~patch_label_aggr.unsqueeze(
            0).eq(invalid_labels).any(dim=0)
        patch_label_aggr = patch_label_aggr[patch_label_mask]
        patch_aggr = patch_aggr[patch_label_mask]

        sampled_patches = torch.cat((sampled_patches, patch_aggr), dim=0)
        sampled_labels = torch.cat((sampled_labels, patch_label_aggr))
    # torch.use_deterministic_algorithms(True)

    return sampled_patches, sampled_labels


def earliest_pose_in_fov(expert, fov_mask, return_idx=False):
    """
    Returns the earliest pose in the field of view mask
    Inputs:
        expert: (batch_sz, T, 2) Expert poses in BEV (original map size)
        fov_mask: (1, 1, height, width) Mask of the field of view
    Outputs:
        S0: (batch_sz, 2) Earliest pose in the field of view
        earliest_idx: (batch_sz) Index of the earliest pose (optional)
        latest_idx: (batch_sz) Index of the latest pose (optional)
    """
    B, T, _ = expert.shape
    H, W = fov_mask.shape[-2:]
    x_coords = expert[:, :, 0].long()  # Shape: (B, T)
    y_coords = expert[:, :, 1].long()  # Shape: (B, T)
    fov_mask = fov_mask.to(x_coords.device)

    # Ensure the pose is within the FOV
    valid_mask = fov_mask[0, 0, x_coords, y_coords] == 1
    valid_idx = torch.where(valid_mask, torch.arange(T).unsqueeze(0).expand(
        B, -1).to(device=valid_mask.device), torch.tensor(T, device=valid_mask.device))
    # Find the earliest valid index
    earliest_idx = valid_idx.min(dim=1).values
    valid_idx[valid_idx==T] = -1 # Set invalid poses to -1
    latest_idx = valid_idx.max(dim=1).values

    no_valid_pose = earliest_idx == T
    earliest_idx = torch.where(no_valid_pose, torch.tensor(
        0, device=valid_mask.device), earliest_idx)

    selected_pose = torch.stack([x_coords[torch.arange(
        B), earliest_idx], y_coords[torch.arange(B), earliest_idx]], dim=1)

    # Set invalid poses to default position (those that didn't find a valid pose)
    selected_pose[no_valid_pose] = torch.tensor(
        [H-1, W//2], dtype=selected_pose.dtype, device=selected_pose.device).repeat(no_valid_pose.sum().item(), 1)
    if return_idx:
        return selected_pose, earliest_idx, latest_idx
    return selected_pose


def gaussian_2d(goals, sigma, H, W):
    """
    Creates a batch of 2D Gaussians centered at (mu_x, mu_y) for each example in the batch
    with standard deviation sigma over a HxW grid.

    Args:
        goals (torch.Tensor): Tensor of shape [B, 2], where B is the batch size,
                            2 corresponds to (mu_x, mu_y) goal coordinates for each sample.
        sigma (float): Standard deviation of the Gaussian.
        H (int): Height of the BEV grid.
        W (int): Width of the BEV grid.

    Returns:
        torch.Tensor: Batch of Gaussian heatmaps of shape [B, 1, H, W].
    """
    B = goals.size(0)
    mu_x = goals[:, 0].view(B, 1, 1).float()  # Shape [B, 1, 1]
    mu_y = goals[:, 1].view(B, 1, 1).float()  # Shape [B, 1, 1]

    x = torch.arange(0, H).view(1, H, 1).float().to(
        goals.device)  # Shape [1, H, 1]
    y = torch.arange(0, W).view(1, 1, W).float().to(
        goals.device)  # Shape [1, 1, W]

    # Create Gaussian heatmap for each goal
    gauss = torch.exp(-((x - mu_x)**2 + (y - mu_y)**2) /
                      (2 * sigma**2))  # Shape [B, H, W]

    return gauss.view(B, 1, H, W)

def balanced_infos_resampling(infos, distances, num_bins=10):
    """
    Distance-balanced resampling of dataset.
    
    Inputs:
        infos: np.array of N info elements (each element could be an array or an object that stores relevant data).
        distances: np.array of N corresponding distance values.
        num_bins: Number of bins to use for partitioning distances (default is 10).
    
    Returns:
        sampled_infos: np.array of resampled infos, balanced based on distances.
    """
    # Ensure distances are a numpy array for easier manipulation
    distances = np.array(distances)
    infos = np.array(infos)

    info_to_distance = {infos[i]: distances[i] for i in range(len(infos))}
    
    # Determine bins for the distances
    bins = np.linspace(min(distances), max(distances), num_bins)
    
    # Group infos into bins based on distances
    bin_indices = np.digitize(distances, bins, right=True)
    bin_indices += 1
    binned_infos = {i: [] for i in range(1, num_bins + 1)}  # Dictionary to store infos per bin
    for info, bin_idx in zip(infos, bin_indices):
        binned_infos[bin_idx].append(info)

    # Calculate total number of samples and the distribution of samples in each bin
    total_samples = len(infos)
    bin_distribution = {i: len(binned_infos[i]) / total_samples for i in binned_infos}
    
    # Desired uniform fraction for each bin
    desired_frac = 1.0 / num_bins
    ratios = {i: desired_frac / (bin_distribution[i] + 1e-3) for i in binned_infos}  # +1e-3 to avoid divide by zero
    
    # Resample infos based on the computed ratios
    sampled_infos = []
    for bin_idx, cur_bin_infos in binned_infos.items():
        # Always preserve the original infos in the bin
        if len(cur_bin_infos) > 0:
            # Determine the factor by which to oversample beyond the original count
            additional_ratio = ratios[bin_idx] - 1.0
            # Calculate how many additional samples to draw for this bin
            num_additional_samples = int(len(cur_bin_infos) * additional_ratio)
            
            # Determine if replacement is necessary when drawing additional samples
            should_replace = num_additional_samples > len(cur_bin_infos)
            
            # Draw additional samples if needed
            if num_additional_samples > 0:
                oversampled = np.random.choice(cur_bin_infos, num_additional_samples, replace=should_replace).tolist()
            else:
                oversampled = []
            
            # Add both original and oversampled infos to the result
            sampled_infos += cur_bin_infos + oversampled

    sampled_infos = np.array(sampled_infos)  # Convert back to numpy array
    sampled_distances = np.array([info_to_distance[info] for info in sampled_infos])

    # Log the new distribution
    print("Total samples before distance-balanced resampling: %s" % (total_samples))
    print('Total samples after distance-balanced resampling: %s' % (len(sampled_infos)))
    
    # Check the new distribution of the sampled infos
    resampled_distances = distances[np.isin(infos, sampled_infos)]
    bin_indices_new = np.digitize(resampled_distances, bins, right=True)
    bin_indices_new += 1
    binned_infos_new = {i: 0 for i in range(1, num_bins + 1)}
    
    for bin_idx in bin_indices_new:
        binned_infos_new[bin_idx] += 1

    bin_distribution_new = {i: binned_infos_new[i] / len(sampled_infos) for i in binned_infos_new}
    
    print('New bin distribution after resampling: %s' % bin_distribution_new)

    # Plot resampled distance distribution
    # plt.figure()
    # plt.hist(resampled_distances, bins=num_bins)
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Resampled Distance Distribution')
    # plt.savefig("resampled_hausdorff_distances.png")
    # plt.close()
    return sampled_infos, sampled_distances

def resize_and_center_crop(images_list, target_sizes_list, intrinsics_list=None):
    """
    Resize and center crop a list of image batches to a target size while preserving aspect ratio.
    
    Each element in images_list is a tensor of shape [B, C, H, W] (with common H and W).
    The function computes the scale factor by taking the ratio of the target dimension 
    to the original and using the larger of the two. This ensures that after resizing, 
    one dimension exactly matches the target size and the other is larger so that a center crop is possible.
    
    If a corresponding intrinsics_list is provided (a list of 3x3 camera intrinsic tensors), 
    then the focal lengths (fx, fy) and principal point (cx, cy) are updated:
      - Multiply fx and fy by the scale.
      - Multiply cx and cy by the scale and then subtract the crop offset.
    
    Args:
        images_list (list of torch.Tensor): List of image batches, each with shape [B, C, H, W].
        target_sizes_list (list of tuple): (target_h, target_w) for the output image dimensions.
        intrinsics_list (list of torch.Tensor, optional): List of 3x3 camera intrinsic matrices, one per batch.
    
    Returns:
        If intrinsics_list is provided:
            (transformed_images_list, transformed_intrinsics_list)
        Otherwise:
            transformed_images_list
    """
    transformed_images_list = []
    transformed_intrinsics_list = [] if intrinsics_list is not None else None

    for i, images in enumerate(images_list):
        B, C, H, W = images.shape
        target_h, target_w = target_sizes_list[i]
        
        # Compute scale factors for height and width
        scale_h = target_h / H
        scale_w = target_w / W
        # Use the larger scale factor so that both dimensions become >= target size.
        scale = max(scale_h, scale_w)
        
        # Compute new dimensions after resizing
        new_h = int(math.ceil(H * scale))
        new_w = int(math.ceil(W * scale))
        
        # Resize the batch
        resized_images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        # Compute center crop offsets
        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        
        # Center crop the images
        cropped_images = resized_images[:, :, top:top+target_h, left:left+target_w]
        transformed_images_list.append(cropped_images)
        
        # Adjust the corresponding camera intrinsics if provided
        if intrinsics_list is not None:
            K = intrinsics_list[i].clone()  # make a copy so we don't modify the original
            # Scale focal lengths
            K[0, 0] = K[0, 0] * scale  # fx
            K[1, 1] = K[1, 1] * scale  # fy
            # Scale principal point and then subtract crop offsets
            K[0, 2] = K[0, 2] * scale - left  # cx (or ux)
            K[1, 2] = K[1, 2] * scale - top   # cy (or uy)
            transformed_intrinsics_list.append(K)

    if intrinsics_list is not None:
        return transformed_images_list, transformed_intrinsics_list
    else:
        return transformed_images_list

if __name__ == '__main__':
    # create_trapezoidal_fov_mask(256, 256)

    # Test resize and center crop
    import cv2
    from PIL import Image
    from torchvision import transforms

    # Aspect ratio check
    # img = Image.open("/robodata/arthurz/Research/lift-splat-map/data/coda_rlang/2d_rect/cam0/3/2d_rect_cam0_3_14.png")

    # Resize and center crop
    img = Image.open("/robodata/arthurz/Research/creste_public/creste_train/front_rgb.jpg")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)

    print(img.shape)
    output_sizes = [(512, 612)]
    output = resize_and_center_crop([img], output_sizes)
    
    cv2.imwrite("input.jpg", img.squeeze().permute(1, 2, 0).numpy() * 255)
    cv2.imwrite("output.jpg", output[0].squeeze().permute(1, 2, 0).numpy() * 255)
    print(f"Wrote input.jpg and output.jpg")
    print("Done")