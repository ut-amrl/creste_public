import os
import time
from os.path import join
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
from PIL import Image
import yaml
# import multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
import torch.multiprocessing as mp
from tqdm import tqdm
import cv2

import gc
import numba
from numba.typed import List

import torch
from torch.utils.data import DataLoader
import creste.utils.depth_utils as du
import creste.utils.elevation_utils as eu
import creste.utils.aggregator_utils as au
import creste.utils.feature_extractor as fe
import creste.utils.visualization as vis
import creste.utils.projection as pj
from creste.utils.visualization import show_elevation_map

from creste.datasets.dataloader import CODaPEFreeModule
import creste.datasets.coda_helpers as ch

from creste.datasets.coda_utils import SEM_ID_TO_COLOR, IGNORE_ELEVATION_CLASSES, SSC_LABEL_DIR, FSC_LABEL_DIR, ELEVATION_LABEL_DIR

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
# Reduce number of torch threads
torch.set_num_threads(6)
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy/15641148#15641148
dataset = None

SAVE_IMAGES = True

WINDOW_SIZE = 50

# Try to set the start method to 'spawn'
try:
    set_start_method('spawn')
except RuntimeError as e:
    print('Start method spawn has already been set.')


class SemanticMap:
    def __init__(
        self,
        grid_height,
        grid_width,
        voxel_size,
        grid_range,
        max_points=131072000,
        max_z=3.0,
        device="cpu"
    ):
        self.device = device
        self.voxel_size = torch.tensor(voxel_size).to(device).float()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_range = torch.tensor(grid_range).to(device).float()
        self.grid_dims = torch.tensor([
            int(grid_height/voxel_size[0]), int(grid_width/voxel_size[1])
        ]).to(device).long()
        self.max_points = max_points

        self.mode = "static"
        self.max_z = max_z
        self.points = None
        self.points_idx = 0
        self.point_dim = 3
        self.static_reduction_mat = None
        self.dynamic_reduction_mat = None
        self.dynamic_points = []

    def set_mode(self, mode):
        assert mode in ["static", "dynamic"], "Invalid mode"
        self.mode = mode

    def add_points(self, points, pose, labels, filter_labels=False):
        """
        points:    (np.ndarray)    N x 3 point cloud
        pose:      (np.ndarray)    4 x 4 pose matrix (lidar2global)
        labels:    (np.ndarray)    N x F labels
        """
        assert points.shape[0] == labels.shape[0], "Points and labels must have the same number of rows"
        if points.shape[0] > 131072:
            print("Exceeded max number of points per frame")
            import pdb; pdb.set_trace()
            return
        assert points.shape[0] <= 131072, "Exceeded max number of points per frame"
        N, F = labels.shape

        # 0 Filter out points without labels
        label_mask = torch.ones((N,), dtype=bool).to(self.device)
        if filter_labels:
            label_mask = labels > 0
        label_mask = label_mask.squeeze()  # [N, 1] -> [N]

        z_mask = points[:, 2] < self.max_z
        # Filter out points with no labels and points above max height
        mask = label_mask & z_mask

        # 1 Transform points to global frame
        points = self.transform_point_cloud(points, pose)  # [N, 3] -> [N, 4]
        filtered_points = points[mask, :3].reshape(-1, 3)
        filtered_labels = labels[mask, :].reshape(-1, F)
        point_features = torch.cat((filtered_points, filtered_labels), axis=-1)

        # 2 Add to either static or dynamic points
        if self.mode == "dynamic":
            self.dynamic_points.append(point_features)
        else:
            N, F = point_features.shape
            if self.points is None:
                self.points = torch.zeros(
                    (self.max_points, F), dtype=torch.float32).to(self.device)
                self.points_idx = 0
            assert N == point_features.shape[0], "Number of points must be the same as the number of labels"
            end_idx = self.points_idx + N
            assert end_idx < self.max_points, "Exceeded max number of points"
            self.points[self.points_idx:end_idx, :] = point_features
            self.points_idx += N

    def compute_reduction_matrix(self, num_components=3):
        """
        Computes pca reduction for the given point cloud
        """
        if self.mode == "dynamic":
            assert len(
                self.dynamic_points) > 0, "No points to compute reduction matrix"

            if self.dynamic_reduction_mat is None:
                points = torch.cat(self.dynamic_points, axis=0)
                features = points[:, 3:]

                self.dynamic_reduction_mat = fe.compute_pca_reduction(
                    features,
                    features.shape[0],
                    3
                )
                rgb_feats = features @ self.dynamic_reduction_mat
                self.dynamic_rgb_min = rgb_feats.min(dim=0).values
                self.dynamic_rgb_max = rgb_feats.max(dim=0).values
            return self.dynamic_reduction_mat, self.dynamic_rgb_min, self.dynamic_rgb_max
        else:
            assert self.points is not None, "No points to compute reduction matrix"

            if self.static_reduction_mat is None:
                points = self.points[:self.points_idx, :]
                features = points[:, 3:]

                # self.reduction_mat, self.rgb_min, self.rgb_max = fe.get_robust_pca(features, 2)

                self.static_reduction_mat = fe.compute_pca_reduction(
                    features,
                    features.shape[0],
                    3
                )
                rgb_feats = features @ self.static_reduction_mat
                self.static_rgb_min = rgb_feats.min(dim=0).values
                self.static_rgb_max = rgb_feats.max(dim=0).values

            return self.static_reduction_mat, self.static_rgb_min, self.static_rgb_max

    def reset_semantic_map(self):
        self.dynamic_points = []
        self.points = None
        self.static_reduction_mat = None
        self.dynamic_reduction_mat = None
        self.mode = "static"
        self.points_idx = 0
        torch.cuda.empty_cache()

    def convert_labels_to_bev(self, point_cloud, label, pix2pt, pt2pix, pose):
        """
        Converts labels from image space to BEV space

        Inputs:
            point_cloud: Nx3 point cloud in LiDAR frame
            label: [Himg, Wimg, F] feature labels in image space
            pix2pt: [4, 3] pose @ pixel to point transformation matrix
            pt2pix: [3, 4] point to pixel transformation matrix @ pose
            pose:   [4, 4] pose matrix
        Outputs:
            valid_image_feats: [Nd, F] valid image features
            valid_lidar_mask: [N, 1] in fov valid lidar mask (Nd will be true)
        """
        Hi, Wi = label.shape[0], label.shape[1]
        # 1 Transform point cloud to image space
        pose_inv = torch.linalg.inv(pose)
        calib_dict = {
            'lidar2camrect': pt2pix @ pose_inv
        }
        
        valid_image_pts, valid_pc_pts, valid_lidar_mask = du.pixels_to_depth(
            point_cloud, calib_dict, Hi, Wi, return_keys=['image_pts', 'pc_pts', 'pc_mask'], IMG_DEBUG_FLAG=False
        )

        valid_image_pts = torch.from_numpy(
            valid_image_pts).long().to(self.device)
        valid_pc_pts = torch.from_numpy(
            valid_pc_pts).long().to(self.device)
        valid_lidar_mask = torch.from_numpy(
            valid_lidar_mask).bool().to(self.device)
        # 3 Extract image feature labels from projected lidar points
        valid_image_feats = label[valid_pc_pts[:,
                                                  1], valid_pc_pts[:, 0], :]

        return valid_image_pts, valid_pc_pts, valid_image_feats, valid_lidar_mask

    def visualize_cloud_at_pose(self, pose):
        """
        Visualizes the point cloud at a given pose
        """
        filepath = "test.png"
        print(f'Visualizing point cloud at pose to {filepath}')
        pc = self.get_pointcloud_from_pose(pose)

        vis.visualize_pc_3d(pc, filepath)

    def get_pointcloud_from_pose(self, pose):
        """
        This function returns an egocentric point cloud given a pose. Assumes robot is centered at center with forward x is down, left y is right and z is up.

        sem_map: (SemanticMap)    The voxel map to get points from
        pose:       (dict)        The pose to get the scene from 'lidar2global' (4x4 np.ndarray), 'ts' (float)
        Output:
            fullpc: (np.ndarray)    N x [3+F] point cloud
        """
        # print("Getting point cloud from pose")

        if self.mode == "dynamic":
            assert type(
                pose) == int, "Dynamic mode requires pose to be an integer"
            fullpc = self.dynamic_points[pose]
        else:
            points = self.points[:self.points_idx, :]

            # Heading aligned egocentric map range
            T_map_to_ref = torch.linalg.inv(pose)

            # 0 Extract points from ego aligned map
            sem_map_xyz_homo = torch.hstack((
                points[:, :self.point_dim],
                torch.ones((points.shape[0], 1)).float().to(self.device)
            ))
            fullpc = (T_map_to_ref @ sem_map_xyz_homo.T).T
            fullpc = torch.hstack(
                (fullpc[:, :self.point_dim], points[:, self.point_dim:]))
            fulloriginalpc = torch.hstack(
                (sem_map_xyz_homo[:, :self.point_dim], points[:, self.point_dim:]))

        min_bound = self.grid_range[:2]  # x_min, y_min
        max_bound = self.grid_range[2:]

        valid_point_mask = torch.all(
            (fullpc[:, :2] < max_bound) & (fullpc[:, :2] >= min_bound), axis=1)
        fullpc = fullpc[valid_point_mask, :]
        fulloriginalpc = fulloriginalpc[valid_point_mask, :]

        # Temporary hack set z value to absolute pc's z value
        # fullpc[:, 2] = fulloriginalpc[:, 2]

        return fullpc

    # Function to transform poses and point clouds
    def transform_point_cloud(self, point_cloud, pose):
        """
        point_cloud:    (np.ndarray)    N x 4 point cloud
        pose:           (np.ndarray)    4 x 4 pose matrix (Usually lidar to map)
        """
        pc = torch.ones((point_cloud.shape[0], 4), device=self.device)
        pc[:, :3] = point_cloud[:, :3]

        transformed_pc = (pose @ pc.T).T
        return transformed_pc

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()


def get_scene_from_pose(sem_map, pose, num_classes):
    """
    This function returns an egocentric voxel map given a pose. Assumes robot is centered at bottom middle
    facing forward.

    sem_map: (SemanticMap)    The voxel map to get points from
    pose:       (dict)        The pose to get the scene from 'lidar2global' (4x4 np.ndarray), 'ts' (float)
    num_classes: (int)        The number of classes in the dataset

    Output:
        feature_map - H x W x F feature map
        Important! Dim0 corresponds to the width and Dim1 corresponds to the height. By default, torch may use dim1 for width and dim0 for height
    """
    # print("Getting scene from pose")

    # 1 Extract points from accumulated map
    fullpc = sem_map.get_pointcloud_from_pose(pose)
    valid_bev_points = fullpc[:, :2]
    valid_bev_labels = fullpc[:, 3:]

    F = valid_bev_labels.shape[1]

    # 2 Extract points from ego aligned map
    min_bound = sem_map.grid_range[:2]  # x_min, y_min
    voxels = torch.floor((valid_bev_points - min_bound) /
                         sem_map.voxel_size).long()

    # Clamp to account for any floating point errors
    maxes = torch.reshape(sem_map.grid_dims - 1, (1, 2))
    mins = torch.zeros_like(maxes)
    voxels = torch.clip(voxels, mins, maxes).long()
    voxels_with_labels = torch.cat(
        (voxels, valid_bev_labels.reshape(-1, F)), axis=-1)

    if num_classes > 0:
        unique_voxels, counts = torch.unique(
            voxels_with_labels, return_counts=True, axis=0)
        unique_voxels = unique_voxels.long()
        feature_map = torch.zeros(
            (sem_map.grid_dims[0], sem_map.grid_dims[1], num_classes)).long()
        feature_map[unique_voxels[:, 0], unique_voxels[:, 1],
                    unique_voxels[:, 2]] += counts
    else:
        # Aggregate features
        ids = voxels_with_labels[:, :2]
        descriptors = voxels_with_labels[:, 2:]
        feature_map = au.aggregate_descriptors(
            ids, descriptors, dims=sem_map.grid_dims, aggregator="GMP"
        )  # H x W x F
        feature_map = feature_map.permute(1, 0, 2)  # [W, H, F] -> [H, W, F]

    return feature_map


def get_elevation_from_pose(sem_map, pose):
    """
    This functions returns an egocentric elevation map given a pose. Assumes robot is centered in the middle
    and forward x is down, left y is right and z is up.

    ________________
    |              |
    |              |
    |     o > y    |
    |     \/       |
    |      x       |
    ________________
    sem_map: (SemanticMap)    The voxel map to get points from
    pose:       (dict)        The pose to get the scene from 'lidar2global' (4x4 np.ndarray), 'ts' (float)
    pc:     (np.ndarray)        N x 3 point cloud query to get elevation values for

    Returns:
        bev_elevation_map: (np.ndarray)    The elevation map in the BEV frame [H x W]
        bev_variance_map: (np.ndarray)      The variance map in the BEV frame [H x W]
    """
    # print("Getting elevation from pose")

    # Heading aligned egocentric map range
    fullpc = sem_map.get_pointcloud_from_pose(pose)

    # 2 TODO: Filter points based on if they are terrain sematic class
    MAP_CONFIG = {
        'map': {
            'width': sem_map.grid_width,        # Map size in meters
            'height': sem_map.grid_height,      # Map size in meters
            'resx': int(sem_map.grid_dims[0]),  # Map size in grid cells
            'resy': int(sem_map.grid_dims[1]),  # Map size in grid cells
            """CODA CONFIG"""
            # 'nlowest_points': 2,
            # 'post_kernel_min_points_per_cell': 3
            """CRESTE CONFIG"""
            'nlowest_points': 2,
            'post_kernel_min_points_per_cell': 3
        },
        'meanz_kernel': {
            'resw': 3,
            'resh': 3,
            'stride': 1
        },
        'threshold': {
            'sky': 2
        }

    }

    # 3 Swap x y axes because mapping process flips them
    full_cupc = fullpc[:, [1, 0, 2, 3]]
    bp = eu.BinningPostprocess(MAP_CONFIG, 'cuda')

    # 4 Filter unlabeled dynamic points
    def find_candidate_ground_points(full_pc):
        # Find points that are not on ground
        valid = None
        for c in IGNORE_ELEVATION_CLASSES:
            if valid is None:
                valid = (full_pc[:, 3] != c)
            else:
                valid &= (full_pc[:, 3] != c)
        return valid

    valid_point_mask = find_candidate_ground_points(full_cupc)
    if valid_point_mask is not None:
        full_cupc = full_cupc[valid_point_mask]

    # 5 Build elevation map from grid of query points
    bp.build_map(full_cupc[:, :3], 'min')
    # [2 x H x W] Channel0 is min elevation, Channel1 is valid_ground
    bev_min_elevation = bp.minz_ground_map.map

    bp.build_map(full_cupc[:, :3], 'max')
    # [2 x H x W] Channel0 is max elevation, Channel1 is valid_ground
    bev_max_elevation = bp.minz_ground_map.map

    # 6 Build variance map from grid of query points
    bp.build_map(full_cupc[:, :3], 'var')
    # [2 x H x W] Channel0 is variance, Channel1 is valid_ground
    bev_variance = bp.minz_ground_map.map

    bev_min_elevation_map = bev_min_elevation[0, :, :]  # Extract elevation map
    bev_max_elevation_map = bev_max_elevation[0, :, :]  # Extract elevation map
    bev_elevation_mask = bev_min_elevation[1, :, :]  # Extract valid ground mask
    bev_variance_map = bev_variance[0, :, :]  # Extract variance map

    # Set invalid ground to min height
    bev_min_elevation_map[bev_elevation_mask == 0] = torch.inf
    bev_max_elevation_map[bev_elevation_mask == 0] = torch.inf
    bev_variance_map[bev_elevation_mask == 0] = 0

    bev_elevation_map = torch.stack(
        (bev_min_elevation_map, bev_max_elevation_map), axis=0)
    bev_elevation_map = bev_elevation_map.permute(1, 2, 0) # [2, H, W] -> [H, W, 2]

    # Flip to match BEV map frame
    bev_elevation_map = torch.flip(bev_elevation_map, [0, 1])
    bev_variance_map = torch.flip(bev_variance_map, [0, 1])

    # import cv2
    # test_elevation_map = bev_elevation_map[0].cpu().numpy()
    # test_elevation_map[test_elevation_map == np.inf] = 0
    # cv2.imwrite('test.png', cv2.normalize(
    #         test_elevation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    # import pdb; pdb.set_trace()
    return bev_elevation_map.cpu().numpy(), bev_variance_map.cpu().numpy()

@numba.njit  # (parallel=True)
def _compute_lower_and_upper_elevation(
    lower_elevs, upper_elevs, proj_class, pc, elevation2, bin_group, sky_thres, gap_thres,
    resx, resy, unique_idxs
):
    """
    lower_elevs, upper_elevs, proj_target will be updated.
    Args:
        lower_elevs: lower elevation map (H x W reshaped into 1D)
        upper_elevs: upper elevation map (H x W reshaped into 1D)
        elevation:  float array of size N, elevation for each point
        proj_class: int array of size N, the projection class for each point
                     0 -> ground, 1 -> ceiling, 2 -> sky
        bin_group: a list of arrays. Each array contains the point indices that belong
                   to a map cell
        sky_thres: threshold to decide if a point is above the projection height
        gap_thres: threshold to decide if a point belongs to the ground or ceiling

    Returns:
        (None)
    """
    # Algorithm to compute the lower and upper elevation
    # From the lowest to highest point
    #     Compute the elevation gap d between the current point and the previous point
    #     If d < gap threshold
    #         Move to the next point
    #     Else
    #         Lower elevation <- previous point
    #         Upper elevation <- current point
    #         Return
    # Set the lower elevation to the current point
    # Set the upper elevation to H_sky

    proj_class[:] = PROJ_SKY
    zs = pc[:, 2]

    for i in range(len(bin_group)):
        point_idxs = bin_group[i]
        ground_elev = elevation[i]
        if not np.isfinite(ground_elev):
            continue
        ground_offsets = zs[point_idxs] - ground_elev
        idxs = np.argsort(ground_offsets)
        elevs_sorted = ground_offsets[idxs]

        prev_elev = 0.0
        gap_found = False
        for j in range(len(elevs_sorted)):
            elev = elevs_sorted[j]
            pidx = point_idxs[idxs[j]]

            assert np.isfinite(elev)

            if elev < 0.0:
                # Ignore points that are below the ground
                continue

            if prev_elev == 0.0 and elev > 0.3:
                # First point is too high above the ground
                break

            if gap_found:
                # Already found the gap, so all the remaining points belong to ceiling or sky
                if elev < sky_thres:
                    proj_class[pidx] = PROJ_CEILING
            elif elev > MIN_OVERHANGING_ELEVATION and elev - prev_elev > gap_thres:
                # First time found the gap
                # The gap can be below the sky threshold or above the sky threshold
                # If the gap is above the sky threshold, then we clamp
                # both the lower and upper elevations to the sky threshold
                lower_elevs[i] = min(prev_elev, sky_thres)
                upper_elevs[i] = min(elev, sky_thres)
                gap_found = True
            else:
                # Haven't found the gap, so this point
                # still belongs to the ground semantics
                proj_class[pidx] = PROJ_GROUND

            prev_elev = elev

        if not gap_found:
            # Haven't found the gap. There are two possibilities:
            # 1. No overhanging points.
            # 2. This object is taller than the sky threshold.
            if prev_elev > 0.0:
                lower_elevs[i] = min(prev_elev, sky_thres)

                # if prev_elev > 2.5:
                #     from bevnet.utils import pointcloud_vis
                #     cam_param = {'scale_factor': 23.93920493691636, 'center': (0.0, 0.0, 0.0),
                #                  'fov': 45.0,
                #                  'elevation': 3.0, 'azimuth': -26.0, 'roll': 0.0}
                #     vis = pointcloud_vis.LaserScanVis(width=512, height=512, interactive=True)
                #     vis.set_camera(cam_param)
                #     vis.draw_points(pc[:, :3], colors=(1.0, 0.0, 0.0), size=2)
                #
                #     m = np.zeros((resy, resx), np.float32)
                #     m.fill(np.nan)
                #     m.reshape(-1)[unique_bins] = lower_elevs
                #     # idx = unique_bins[i]
                #     # m.reshape(-1)[idx] = min(prev_elev, sky_thres)
                #     vis.draw_mesh_grid(m, ~np.isnan(m), 0.4)
                #     vis.show()

            upper_elevs[i] = sky_thres


def compute_lower_and_upper_elevation(pc, elevation, index, resx, resy, sky_thres=2, gap_thres=0.1):
    """
    This function computes the lower and upper elevation maps given a point cloud and an elevation map.
    It will compute the lower and upper elevation maps according to the algorithm in the paper.

    pc: (np.ndarray)    The point cloud to get points from [N x 3]
    elevation: (np.ndarray) The elevation map to get the lower and upper elevation maps from [H x W]
    index: (np.ndarray) The indices to the BEV map for each point [N]
    resx: (int) The number of cells in the x direction
    resy: (int) The number of cells in the y direction
    """
    print("Computing lower and upper elevation")

    assert np.all(index < resx*resy), "Index out of max bounds"
    assert np.all(index >= 0), "Index out of min bounds"

    unique_idxs, bin_idxs = np.unique(index, return_inverse=True)

    # bin_idxs contains the bin index of each point
    # group bin_idxs such that bin_group[i] contains the indices of points with bin_idx i
    # to obtain the actual grid location of a bin, use unique_idxs[i]
    lookup = np.argsort(bin_idxs)
    sorted_bin_idxs = bin_idxs[lookup]
    boundaries = np.nonzero(np.diff(sorted_bin_idxs) > 0)[0]

    bin_group = List()
    last_b = 0
    for b in boundaries:
        bin_group.append(lookup[last_b: b + 1])
        last_b = b + 1
    bin_group.append(lookup[last_b:])

    # Compute the lower and upper elevation maps
    lower_elevs = np.zeros_like(len(bin_group), np.float32)
    upper_elevs = np.zeros_like(len(bin_group), np.float32)
    proj_class = np.zeros(len(index), np.int64)

    lower_elevs.fill(np.nan)
    upper_elevs.fill(np.nan)

    elevation2 = elevation[unique_idxs].copy()
    elevation2[~np.isfinite(elevation2)] = np.nan

    # Run algorithm to compute lower and upper elevations
    _compute_lower_and_upper_elevation(
        lower_elevs, upper_elevs, proj_class, pc, elevation2, bin_group, sky_thres, gap_thres,
        resx, resy, unique_idxs
    )


def save_elevation_to_file(elevation_map, variance_map, save_root_dirs, seq, frame, show_vis=False):
    """
    elevation_map - (np.ndarray)    The elevation map to save to file [2, height x width]
    variance_map - (np.ndarray)      The variance map to save to file [height x width]
    filepath - (str)                The path to save the scene to
    """
    elevation_map = elevation_map.astype(np.float32)
    variance_map = variance_map.astype(np.float32)

    # Save as binary file
    elevation_dir = join(save_root_dirs[0], seq)
    if not os.path.exists(elevation_dir):
        os.makedirs(elevation_dir)
    elevation_path = join(elevation_dir, f'{str(frame)}.bin')

    variance_dir = join(save_root_dirs[2], seq)
    if not os.path.exists(variance_dir):
        os.makedirs(variance_dir)
    variance_path = join(variance_dir, f'{str(frame)}.bin')

    # Convert scene to BEV map frame
    # Uncomment this to save elevation and variance maps
    elevation_map.tofile(elevation_path)
    variance_map.tofile(variance_path)
    print(f'Saved elevation to {elevation_path}')

    if show_vis:  # Save scene visualization
        elevation_img_dir = join(save_root_dirs[1], seq)
        if not os.path.exists(elevation_img_dir):
            os.makedirs(elevation_img_dir)
        elevation_img_path = join(elevation_img_dir, f'{str(frame)}.jpg')
        min_elevation_img = show_elevation_map(elevation_map[:, :, 0], color_scale="relative")
        offset_elevation = elevation_map[:, :, 1] - elevation_map[:, :, 0]
        offset_elevation_img = show_elevation_map(offset_elevation, color_scale="relative")
        combined_elevation_img = np.concatenate(
            (min_elevation_img, offset_elevation_img), axis=1)
        cv2.imwrite(elevation_img_path, combined_elevation_img)

        variance_img_dir = join(save_root_dirs[3], seq)
        if not os.path.exists(variance_img_dir):
            os.makedirs(variance_img_dir)
        variance_img_path = join(variance_img_dir, f'{str(frame)}.jpg')
        show_elevation_map(variance_map, color_scale="relative", filepath=variance_img_path)


def save_scene_to_file(
    sem_map, scene, save_root_dirs, seq, frame, bev_scene=False, show_vis=False, reducer="argmax"
):
    """
    scene - (np.ndarray)    The voxelized scene to save to file [height x width x num_classes]
    filepath - (str)        The path to save the scene to
    """
    # Save as binary file
    label_dir = join(save_root_dirs[0], seq)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    label_path = join(label_dir, f'{str(frame)}.bin')

    if bev_scene:
        scene = torch.flip(scene, [0, 1])

    if show_vis:  # Save scene visualization
        if reducer == "argmax":
            sem_map = torch.argmax(scene, axis=-1)
            id2rgb = torch.from_numpy(
                np.array(SEM_ID_TO_COLOR, dtype=torch.uint8))
            rgb_map = id2rgb[sem_map]  # Save original class mappings
        elif reducer == "pca":
            H, W, F = scene.shape
            features = scene.reshape(H*W, F)
            reduction_mat, rgb_min, rgb_max = sem_map.compute_reduction_matrix()

            rgb_feats = features @ reduction_mat
            rgb_feats = (rgb_feats - rgb_min) / (rgb_max - rgb_min)

            # Convert to HxWx3 8bit rgb
            rgb_map = rgb_feats.reshape(H, W, 3)
            rgb_map = (rgb_map*255).cpu().numpy().astype(np.uint8)
        else:
            raise ValueError(f"Invalid reducer {reducer}")

        scene_img = Image.fromarray(rgb_map[:, :, ::-1])  # Convert to RGB
        scene_img_dir = join(save_root_dirs[1], seq)
        if not os.path.exists(scene_img_dir):
            os.makedirs(scene_img_dir)

        scene_img_path = join(scene_img_dir, f'{str(frame)}.jpg')
        scene_img.save(scene_img_path)
    # scene_img.save("test.png")

    # Convert scene to np array
    scene_np = scene.float().cpu().numpy()
    scene_np.tofile(label_path)
    # print(f'Saved scene to {label_path}')


def init_worker(shared_sem_map, shared_dataset):
    global dataset

    # Initialize or load sem_map and dataset here
    dataset = shared_dataset  # Placeholder for actual loading/initialization function


def process_single_frame(inputs):
    global SAVE_IMAGES
    # global dataset
    dataset, sem_map, rewind_idx, last_idx, device, task_save_paths = inputs

    batch = dataset[rewind_idx]
    batch_pose = batch['pose'][0].to(device)
    batch_sequence = str(batch['sequence'][0].item())
    batch_frame = str(batch['frame'][0].item())

    if SSC_LABEL_DIR in task_save_paths:
        num_classes = dataset.get_num_labels(task=SSC_LABEL_DIR)
        scene = get_scene_from_pose(
            sem_map, batch_pose, num_classes=num_classes)
        save_scene_to_file(
            sem_map,
            scene,
            task_save_paths[SSC_LABEL_DIR],
            batch_sequence,
            batch_frame,
            bev_scene=True,
            show_vis=SAVE_IMAGES,
            reducer="argmax"
        )

    if FSC_LABEL_DIR in task_save_paths:
        # Static sceneVisualize point lcoud in 3d
        scene = get_scene_from_pose(sem_map, batch_pose, num_classes=0)
        save_scene_to_file(
            sem_map,
            scene,
            task_save_paths[FSC_LABEL_DIR][0:2],
            batch_sequence,
            batch_frame,
            bev_scene=True,
            show_vis=SAVE_IMAGES,
            reducer="pca",
        )

        sem_map.set_mode("dynamic")
        scene = get_scene_from_pose(sem_map, int(
            rewind_idx-last_idx), num_classes=0)
        save_scene_to_file(
            sem_map,
            scene,
            task_save_paths[FSC_LABEL_DIR][2:4],
            batch_sequence,
            batch_frame,
            bev_scene=True,
            show_vis=SAVE_IMAGES,
            reducer="pca",
        )
        sem_map.set_mode("static")

    if ELEVATION_LABEL_DIR in task_save_paths:
        elevation, variance = get_elevation_from_pose(sem_map, batch_pose)
        # Don't show visualization when returning the elevation for each point
        save_elevation_to_file(
            elevation,
            variance,
            task_save_paths[ELEVATION_LABEL_DIR],
            batch_sequence,
            batch_frame,
            SAVE_IMAGES
        )
        # TODO: Save elevation for each point ineach of bev map
        # save_elevation_to_file(elevation, out_dir, str(batch['seq']), str(batch['frame']), True,args.vis)


def process_chunk(inputs, gpu_id):
    # global dataset
    task_id, dataset, sem_map_params, frame_range, save_range, feat_type, label_key, device, task_save_paths = inputs
    # print("Setting device to ", device)
    # torch.cuda.set_device(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"Task {task_id} running on device {device}")

    feature_types = ["fimg_label", "fimg_pred"]
    filter_labels = feat_type not in feature_types

    sem_map = SemanticMap(
        **sem_map_params, device=device
        # grid_height, grid_width, voxel_size, grid_range, max_z=3.0, device=device
    )

    if int(frame_range[1] - frame_range[0]) > 600:
        print("Frame range too large, splitting into smaller chunks")
        import pdb; pdb.set_trace()

    for i in range(frame_range[0], frame_range[1]):
        batch = dataset[i]
        b=0
        seq, frame = int(batch['sequence'][b]), int(batch['frame'][b])

        # print(f'Processing sequence {seq}, frame {frame}')
        # Load feature prediction
        # print(f'Processing sequence {seq}, frame {frame}')
        point_cloud, label = batch['point_cloud'][b], batch[label_key][b]
        if feat_type != "geometric":
            label = label.permute(1, 2, 0)
        else:
            label = label.unsqueeze(-1)
        p2p, pt2pix = batch['p2p_in'][b], batch['pt2pix_in'][b]
        pose = batch['pose'][b]
        immovable_mask = batch['immovable_depth_label'][b]

        # Move to device
        point_cloud = point_cloud.to(device)
        label = label.to(device)
        p2p = p2p.to(device)
        pt2pix = pt2pix.to(device)
        pose = pose.to(device)
        immovable_mask = immovable_mask.to(device)

        # Transform features to egocentric bev space
        if feat_type in feature_types:
            _, _, label, pc_mask = sem_map.convert_labels_to_bev(
                point_cloud, label, p2p, pt2pix, pose)

            # static_cloud = point_cloud[pc_mask, :]
            # dynamic_cloud = point_cloud[~pc_mask, :]
            # static_label = label[pc_mask, :]

            # Split point clud into static and dynamic clouds
            static_mask_full = torch.logical_and(
                immovable_mask.squeeze(), pc_mask)
            dynamic_mask_full = torch.logical_and(
                ~immovable_mask.squeeze(), pc_mask)
            static_mask_part = immovable_mask[pc_mask].squeeze()
            dynamic_mask_part = ~immovable_mask[pc_mask].squeeze()

            # Decompose point cloud into static and dynamic clouds adn features
            static_cloud = point_cloud[static_mask_full, :]
            dynamic_cloud = point_cloud[dynamic_mask_full, :]
            static_label = label[static_mask_part, :]
            dynamic_label = label[dynamic_mask_part, :]
            
            # Save dynamic clouds for each frame
            sem_map.set_mode("dynamic")
            sem_map.add_points(dynamic_cloud, pose,
                            dynamic_label, filter_labels=filter_labels)
            sem_map.set_mode("static")

            # # Save point cloud as pcd
            # vis.numpy_to_pcd(static_cloud, "static.pcd")
            # vis.numpy_to_pcd(dynamic_cloud, "dynamic.pcd")
        else:  # Only valid with ground truth labels
            image_pts, pc_pts, label, pc_mask = sem_map.convert_labels_to_bev(
                point_cloud, label, p2p, pt2pix, pose) # 0 dynamic, 1 static

            # static_mask = torch.zeros_like(pc_mask)
            # mask_ids = torch.arange(0, pc_mask.shape[0], device=device)
            # mask_ids = mask_ids[immovable_mask[image_pts[:, 1], image_pts[:, 0]]]
            # static_mask[pc_mask] = immovable_mask[image_pts[:, 1], image_pts[:, 0]]
            # label_mask = torch.logical_and(pc_mask, static_mask)

            # Use the sam2 label to isolate static loud
            # label_mask = label>0
            # static_cloud = point_cloud[pc_mask, :]
            # dynamic_cloud = None
            # static_label = label_mask
            # dynamic_label = None

            #
            static_cloud = point_cloud
            dynamic_cloud = None
            static_label = torch.ones_like(pc_mask).unsqueeze(-1)
            dynamic_label = None

        # Add the points to running voxel map
        sem_map.add_points(static_cloud, pose, static_label,
                        filter_labels=filter_labels)

    # Save the scene for each frame
    # with tqdm(total=save_range[1]-save_range[0]) as pbar:
    for i in range(save_range[0], save_range[1]):
        process_single_frame(
            (dataset, sem_map, i, save_range[0], device, task_save_paths))
        # pbar.update(1)

    print("Finished processing save range ", save_range)
    sem_map.reset_semantic_map()
    torch.cuda.empty_cache()

    return f"Task {task_id} completed on GPU {gpu_id}"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build semantic map from point clouds')
    # parser.add_argument('--cfg', type=str, default='./configs/dataset/coda_osam_joint_ds2.yaml')
    parser.add_argument(
        '--cfg', type=str, default='./configs/dataset/distillation/creste_pefree_dinov2.yaml')
    parser.add_argument('--model_type', type=str, default='dino_vitb8')
    parser.add_argument('--out_dir', type=str,
                        default='./postprocess_rlang/build_map_outputs')
    parser.add_argument('--feat_type', type=str, default='fimg_pred',
                        help='[fimg_pred, fimg_label, geometric, semantic]')
    parser.add_argument('--tasks', nargs='+', type=str,
                        help='List of tasks to save. Pass a space separated list "3d_fsc elevation 3d_ssc"')
    parser.add_argument('--skip_factor', type=int, default=5,
                        help='Number of frames to skip when checking for continuity, defaults to 5')
    parser.add_argument('--vis', action='store_true',
                        help="Saves visualization of map, defaults to False")
    parser.add_argument('--feat_dir', type=str, default="",
                        help="Feature directory to load features from if specified, only necessary for fimg_pred and fimg_label")
    parser.add_argument('--feat_dim', type=int, default=64,
                        help="Feature dimension for fimg_pred and fimg_label")
    parser.add_argument('--seq_list', nargs='+', type=int, default=None, help="Sequence to convert, default is all")
    args = parser.parse_args()

    return args

def worker(gpu_id, task_queue, results):
    """
    Worker process to handle tasks for a specific GPU.
    """
    while not task_queue.empty():
        try:
            task_args = task_queue.get_nowait()
        except Exception:
            break

        result = process_chunk(task_args, gpu_id)
        results.append(result)

def main(args):
    global SAVE_IMAGES
    # Parameters
    out_dir = args.out_dir
    feat_type = args.feat_type
    skip_factor = args.skip_factor
    model_type = args.model_type
    feat_dir = args.feat_dir
    feat_dim = args.feat_dim
    SAVE_IMAGES = args.vis
    seq_list = args.seq_list

    if feat_type == "fimg_pred":
        print(
            f'------ Saving predicted features for model type {model_type} ------')
        assert os.path.exists(
            feat_dir), f'Feature directory {feat_dir} does not exist'
        save_subdir = '/'.join(feat_dir.split('/')[-6:-2])
        out_dir = join(out_dir, save_subdir)
    elif feat_type == "fimg_label":
        print(
            f'------ Saving label features for model type {model_type} ------')
        out_dir = join(out_dir, model_type, str(feat_dim))
    elif feat_type == "semantic":
        out_dir = join(out_dir, "groundtruth")
    elif feat_type == 'geometric':
        out_dir = join(out_dir, "geometric")
    else:
        raise ValueError(f"Invalid feature type {feat_type}")

    task_save_paths = {}
    for task in args.tasks:
        if task == FSC_LABEL_DIR:
            task_save_paths[task] = [
                join(out_dir, task, "static", "labels"),
                join(out_dir, task, "static", "images"),
                join(out_dir, task, "dynamic", "labels"),
                join(out_dir, task, "dynamic", "images")
            ]
        elif task == ELEVATION_LABEL_DIR:
            task_save_paths[task] = [
                join(out_dir, task, "labels"),
                join(out_dir, task, "images"),
                join(out_dir, "variance", "labels"),
                join(out_dir, "variance", "images")
            ]
        elif task == SSC_LABEL_DIR:
            task_save_paths[task] = [
                join(out_dir, task, "labels"),
                join(out_dir, task, "images")
            ]
        else:
            raise ValueError(f"Invalid task {task}")

    print(f'------ Saving to {out_dir} ------')
    assert os.path.exists(args.cfg), f'Config file {args.cfg} does not exist'
    with open(args.cfg, 'r') as f:
        cfg_file = yaml.safe_load(f)

    grid_height, grid_width, grid_range = cfg_file['map_size'][0], cfg_file['map_size'][1], \
        np.array(cfg_file['map_range'])[[0, 1, 3, 4]].tolist()
    voxel_size = cfg_file['voxel_size']

    # sem_map = SemanticMap(
    #     grid_height, grid_width, voxel_size, grid_range, max_z=3.0, device=device
    # )

    fload_keys = ['sequence', 'frame',
                  'pose', "point_cloud", "immovable_depth_label"]
    if feat_type == "fimg_pred":
        fload_keys.append("fimg_pred")
        label_key = "fimg_pred"
    elif feat_type == "fimg_label":
        fload_keys.append("fimg_label")
        label_key = "fimg_label"
    elif feat_type == "semantic":
        label_key = "3d_ssc_label"
    elif feat_type == "geometric":
        label_key = "immovable_depth_label"
    else:
        raise ValueError(f"Invalid feature type {feat_type}")

    # label_key = "fimg_label"
    # label_key = "fimg_pred"
    # Initialize the dataset and dataloader
    print("Loading dataset with cfg ", cfg_file)
    # Exclude sequences not specied in user list
    default_skip_sequences = cfg_file['skip_sequences']
    all_sequences = ch.get_available_sequences(cfg_file['root_dir'])
    if seq_list is not None and len(seq_list) > 0:
        default_skip_sequences = [
            seq for seq in all_sequences if (seq in default_skip_sequences) or (seq not in seq_list)
        ]
    codatamodule = CODaPEFreeModule(
        cfg_file,
        views=1,
        batch_size=1,
        num_workers=4,
        fload_keys=fload_keys,
        sload_keys=['p2p'],
        # skip_sequences=default_skip_sequences,
        skip_sequences=[],
        # skip_sequences=[8, 14, 15, 22],
        # skip_sequences=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22],
        # fimg_pred_dir=feat_dir,
        use_global_pose=True
    )
    codatamodule.setup("full")
    dataloader = codatamodule.full_dataloader()
    dataset = dataloader.dataset

    print("len dataset", len(dataset))

    # Precompute the start and end of each sequence in the dataset, initialize the following
    """
    dataset = dataloader instance
    frame_range = [start_idx, end_idx]
    save_range = [start_save_idx, end_save_idx]
    skip_factor
    """

    frame_infos = dataset.idx_to_sample.astype(str)

    # """begin new code"""
    # # Sanity check that frame infocs matches with directory paths
    # check_dir = "/scratch/arthurz/Datasets/CompleteNet/data/coda_pefree/dinov2_vitb14_reduced/cam0"
    # manual_frame_infos = []
    # for seq in range(0, 22):
    #     if seq in [8, 14, 15]:
    #         continue
    #     seq_dir = join(check_dir, str(seq))
    #     frames_list = ch.get_dir_frame_info(seq_dir, ext="npy", short=True)
    #     manual_frame_infos.extend([(seq, frame) for frame in frames_list])

    # """end new code"""

    # Compute when the sequence in column 0 chagnes
    seq_changes = np.where(
        frame_infos[:, 0] != np.roll(frame_infos[:, 0], 1))[0]

    # Compute the start and end of each sequence
    seq_ranges = []
    for i in range(1, len(seq_changes)):
        seq_ranges.append((seq_changes[i-1], seq_changes[i]))
    if len(seq_ranges) == 0:
        seq_ranges.append((0, len(frame_infos)))
    else:
        seq_ranges.append((seq_changes[-1], len(frame_infos)))

    # Maximum number of frames to save in a chunk
    # max_frames_per_chunk = 800 # 15000 seconds
    max_frames_per_chunk = 400  # 15000 seconds

    # Buffer to add to the start and end of each save range
    # buffer = 100 # 500 seconds
    buffer = 50

    # List to hold the save ranges for each sequence
    chunk_ranges = []
    save_ranges = []

    # Divide the frames in each sequence into chunks and set the save range
    for start, end in seq_ranges:
        num_frames = end - start
        num_chunks = num_frames // max_frames_per_chunk

        for i in range(num_chunks):
            save_start = start + i * max_frames_per_chunk
            save_end = min(save_start + max_frames_per_chunk, end)

            chunk_start = max(start, save_start - buffer)
            chunk_end = min(end, save_end + buffer)

            chunk_ranges.append((chunk_start, chunk_end))
            save_ranges.append((save_start, save_end))

        # Handle the last chunk, which may be smaller than max_frames_per_chunk
        if num_frames % max_frames_per_chunk != 0:
            save_start = start + num_chunks * max_frames_per_chunk
            save_end = end

            chunk_start = max(start, save_start - buffer)
            chunk_end = min(end, save_end + buffer)

            chunk_ranges.append((chunk_start, chunk_end))
            save_ranges.append((save_start, save_end))

    print("CHUNK RANGES TO USE ", chunk_ranges)
    print("SAVE RANGES TO USE ", save_ranges)
    
    sem_map_params = {
        "grid_height": grid_height,
        "grid_width": grid_width,
        "voxel_size": voxel_size,
        "grid_range": grid_range,
        # Maximum number of points to store in the map
        "max_points": int(131072*(max_frames_per_chunk+2*buffer)),
        "max_z": 2.0
    }

    # Parallelize processing each chunk
    num_devices = torch.cuda.device_count()
    cuda_device_list = [f'cuda:{i}' for i in range(num_devices)]
    cuda_device_args = np.tile(
        np.array(cuda_device_list), len(chunk_ranges)//num_devices + 1)
    task_args = [
        (chunk_idx, dataset, sem_map_params, chunk_ranges[chunk_idx], save_ranges[chunk_idx],
         feat_type, label_key, cuda_device_args[chunk_idx], task_save_paths)
        for chunk_idx in range(len(chunk_ranges))
    ]
    process_chunk(task_args[0], 0)

    # Precreate the save directories
    outdir_list = []
    for task in task_save_paths:
        for subdir in task_save_paths[task]:
            seq_list = ch.get_available_sequences(cfg_file['root_dir'])
            seq_list = [seq for seq in seq_list if seq not in cfg_file['skip_sequences']]
            for seq in seq_list:
                seq_dir = join(subdir, str(seq))
                if not os.path.exists(seq_dir):
                    os.makedirs(seq_dir)

    # arg_splits = np.array_split(task_args, num_devices)
    # task_args = task_args[int(17+9+27):]
    # task_args = task_args[]

    # Use torch multiprocessing to process chunks in parallel
    # Create a Pool of worker processes
    # with Pool(processes=num_devices) as pool: # For some reason these needs to be 2 times number of devices
    #     # Make progress bar
    #     with tqdm(total=len(task_args)) as pbar:
    #         for _ in pool.imap(process_chunk, task_args):
    #             pbar.update()

    # process_chunk(task_args[0], 0)
    # for task_arg in task_args:
    #     process_chunk(task_arg, 0)
    # Create a task queue and add all tasks
    task_queue = mp.Queue()
    for task_arg in task_args:
        task_queue.put(task_arg)

    # Shared list to store results
    manager = mp.Manager()
    results = manager.list()

    # Create worker processes
    num_gpus = 4
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, task_queue, results))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Print results
    for result in results:
        print(result)

    print("Dataset Processing Complete...")
    # At this point, sem_map contains the voxelized point cloud
    # Depending on the use case, you might want to convert this to a dense grid or save it to a file.


if __name__ == '__main__':
    args = parse_args()
    main(args)
