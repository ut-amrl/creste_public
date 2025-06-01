"""
This scripts builds the ground truth label map using mask labels extracted from SAM. It performs
iterative greedy maximum intersection label assignment to assign the most probable mask label between images.

It produces ground truth label maps for each frame in the dataset. The labels are guaranteed to be continous
with 0 corresponding to unlabeled BEV patches
"""

import os
from os.path import join
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import glob
import copy
import re
from pathlib import Path
import json
from joblib import Parallel, delayed
from typing import List, Union, Tuple, Dict, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
import cv2
import open3d as o3d

# CUDA RELATED IMPORTS
import torch
from transformers import pipeline
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from cuml.cluster import DBSCAN
from scipy.sparse import coo_matrix

from scripts.preprocessing.build_dense_depth import (get_frames_from_json, load_calib, transform_pc_frames)
from creste.datasets.coda_utils import (CAMERA_DIR, CAMERA_SUBDIRS, POINTCLOUD_DIR, POINTCLOUD_SUBDIR, frame2fn, fn2frame, POSES_DIR, POSES_SUBDIRS, SAM_DYNAMIC_LABEL_MAP, SAM_DYNAMIC_COLOR_MAP)
import creste.utils.utils as utils
import creste.utils.visualization as vis
import creste.utils.geometry as geo
from creste.utils.projection import (pixels_to_depth, get_pts2pixel_transform, get_pixel2pts_transform, cam2world, points2voxels)
from creste.datasets.coda_utils import *
import creste.datasets.coda_helpers as ch


os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

# print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
# Reduce number of torch threads
torch.set_num_threads(6)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build semantic map from point clouds')
    parser.add_argument(
        '--indir', type=str, default='./data/creste', help="Path to root directory")
    parser.add_argument(
        '--outdir', type=str, default='./data/creste/sam2_map', help="Path to output directory")
    parser.add_argument('--skip_factor', type=int, default=5,
                        help='Number of frames to skip for aggregation')
    parser.add_argument('--camid', type=str, default='cam0', help='SAM Mask')
    parser.add_argument('--img_ds', type=int, default=2,
                        help='Downsample factor for images')
    parser.add_argument('--skip_sequences', nargs='+', type=int, default=[],
                        help='List of sequences to skip')
    parser.add_argument('--horizon', type=int, default=40,
                        help='Number of frames to aggregate for movability mask computation')
    parser.add_argument('--horizon_ref', type=int, default=0,
                        help='Reference frame for pose transformation')
    args = parser.parse_args()
    return args


def load_sam_horizon(sam_paths, img_shape, mask_type):
    """This function loads a horizon from npy files"""
    sam_horizon = []
    for sam_path in sam_paths:
        # Silent error handling for missing files
        if sam_path[0] is None:
            H, W = img_shape
            sam_mask = np.zeros((2, H, W))
            sam_horizon.append(sam_mask)
            continue

        # Loads a mask from npy file [H, W]
        sam_mask = np.load(sam_path[0]) # TODO: Why is this a np.uint8 type?
        if mask_type == "dynamic":
            json_path = sam_path[0].replace(
                "npy", "json").replace("dynamic", "json")
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Create mapping from instance id to class id
            instance_to_class_id = {int(instance_id): SAM_DYNAMIC_LABEL_MAP.get(instance_dict['class_name'], 0) for instance_id, instance_dict in json_data['labels'].items()}
            instance_to_class_id.update({0: 0})  # Add background class

            # Remap instance ids to class ids and stack to sam_mask
            sam_class_mask = np.vectorize(instance_to_class_id.get)(sam_mask)
            sam_mask = np.stack([sam_mask, sam_class_mask], axis=0)
        else:
            sam_mask = sam_mask[None, ...]

        # sam_mask = np.fromfile(sam_path[0], dtype=np.uint8).reshape(img_shape)
        sam_horizon.append(sam_mask)
    return sam_horizon

# def load_mvlabel_horizon(mv_paths, img_shape):
#     """This function loads a horizon of movability masks"""
#     mv_horizon = []
#     for mv_path in mv_paths:
#         mv_mask = cv2.imread(mv_path[0], cv2.IMREAD_GRAYSCALE)
#         mv_mask = cv2.resize(mv_mask, img_shape[::-1])
#         mv_horizon.append(mv_mask)
#     return mv_horizon


def load_pc_horizon(pc_paths):
    pc_horizon = []
    for pc_path in pc_paths:
        pc_np = np.fromfile(pc_path[0], dtype=np.float32).reshape(POINTS_PER_SCAN, -1)
        pc_horizon.append(pc_np[:, :3])
    return pc_horizon


def load_depth_horizon(depth_paths, img_shape):
    """This function loads a depth image into meters"""
    depth_horizon = []
    for depth_path in depth_paths:
        depth_img = cv2.imread(depth_path[0], cv2.IMREAD_ANYDEPTH)
        depth_img = depth_img.astype(np.float32) / 1000
        depth_horizon.append(cv2.resize(depth_img, img_shape[::-1]))
    return depth_horizon


def load_p2p(calib_dict, poses=None, ds=2):
    calib_dict = copy.deepcopy(calib_dict)

    # Scale projection matrices for backprojection
    calib_dict['K'][:2, :] = calib_dict['K'][:2, :] / ds
    calib_dict['P'][:2, :] = calib_dict['P'][:2, :] / ds

    # Transform to orientation stable reference frame
    p2p = torch.tensor(
        get_pixel2pts_transform(calib_dict)
    ).float().unsqueeze(0)

    if poses is not None:
        p2p = torch.matmul(poses, p2p)

    return p2p


def compute_label_mapping(bev_maps, ignore=0):
    """
    Compute a label mapping based on intersections between two BEV maps.

    Args:
        bev_maps (torch.Tensor): Tensor of shape (B, H, W) containing BEV pixel labels.
        ignore (int): Label to ignore during processing, typically the background.

    Returns:
        dict: Mapping from labels in the anchor view to labels in the opposing view.
    """
    assert bev_maps.size(0) == 2, "bev_maps should contain exactly two maps"

    anchor_map = bev_maps[0]
    opposing_map = bev_maps[1]

    # Extract unique labels excluding the ignore label
    unique_labels_anchor = torch.unique(anchor_map[anchor_map != ignore])
    unique_labels_opposing = torch.unique(opposing_map[opposing_map != ignore])

    # Dictionary to hold the mapping from anchor to opposing labels
    # label_mapping = torch.zeros((len(unique_labels_anchor)), dtype=torch.long).to(bev_maps.device)
    label_mapping = {}

    # Loop over each unique label in the opposing map
    for label in unique_labels_opposing:
        opposing_mask = (opposing_map == label)

        # Compute intersections with each label in the anchor map
        intersections = torch.tensor([
            (opposing_mask & (anchor_map == anchor_label)
             ).float().sum()  # Intersection count
            for anchor_label in unique_labels_anchor
        ])

        if intersections.sum() == 0:
            continue  # No intersections, skip to the next label

        # Find the label in the anchor map with the maximum intersection
        max_label_idx = intersections.argmax()
        corresponding_label = unique_labels_anchor[max_label_idx].item()

        # Add the mapping from opposing label to the corresponding label in the anchor map
        label_mapping[label.item()] = corresponding_label

    return label_mapping


def visualize_bev_map(bev_map, cmap=None, filepath=None):
    """
    Plot bev_map where the color of each xy pixel value is denoted by the label

    Inputs:
    bev_map: torch.Tensor (B, H, W)
    cmap: torch.Tensor (N, 4) where N is the number of labels and 4 is the RGBA values
    """
    _, H, W = bev_map.shape
    bev_map = bev_map[0].cpu().numpy()
    # assert len(np.unique(bev_map))==np.max(bev_map)+1, "Labels are not contiguous"

    num_labels = len(np.unique(bev_map))
    if cmap is None:
        cmap = generate_cmap(num_labels)
        cmap[0] = [0, 0, 0, 0]  # Set background to transparent
    else:
        cmap = cmap.cpu().numpy()
    bev_map_rgb = cmap[bev_map]
    bev_map_rgb = (bev_map_rgb * 255).astype(np.uint8)

    if filepath is not None:
        cv2.imwrite(filepath, bev_map_rgb)

    return bev_map_rgb


def merge_maps(bev_maps, color_map, label_mapping, ignore=0):
    """
    Merge two BEV maps using a label mapping.

    Args:
        bev_maps (torch.Tensor): Tensor of shape (2, H, W) containing BEV pixel labels.
        label_mapping (dict): Mapping from labels in the anchor view to labels in the opposing view.
        color_map (torch.Tensor): Color map for visualization. [Nx4]
        label_mapping (Dict)[int, int]: Mapping from labels in the opposing view to labels in the anchor view.
        ignore (int): Label to ignore during processing, typically the background.

    Returns:
        anchor_map (torch.Tensor): Merged BEV map with labels from the anchor view. [1, H, W]
        merged_color_map (torch.Tensor): Merged color map for visualization. [Nx4]
    """
    assert bev_maps.size(0) == 2, "bev_maps should contain exactly two maps"

    anchor_map = bev_maps[0]
    opposing_map = bev_maps[1]

    # Create a copy of the opposing map
    opposing_remap = torch.zeros_like(opposing_map)

    # Remap labels in the opposing view to the anchor view
    unmapped_labels = []
    for oppose_idx, anchor_idx in label_mapping.items():
        opposing_remap[opposing_map == oppose_idx] = anchor_idx

    # Remap unused labels to new labels outside of current label range
    unused_labels = torch.tensor([
        oppose_idx.item() for oppose_idx in torch.unique(opposing_map)
        if ((oppose_idx.item() not in label_mapping.keys()) and (oppose_idx.item() != 0))
    ])
    for idx, unused_label in enumerate(unused_labels):
        opposing_remap[opposing_map ==
                       unused_label] = anchor_map.max() + idx + 1
        # print(f"Remapping {unused_label} to {anchor_map.max() + idx + 1}")

    if len(unused_labels) > 0:
        unused_color_map = vis.generate_cmap(len(unused_labels), use_th=True)
        merged_color_map = torch.cat((color_map, unused_color_map), dim=0)
    else:
        merged_color_map = color_map

    valid_mask = (anchor_map == 0) & (opposing_remap != 0)
    anchor_map[valid_mask] = opposing_remap[valid_mask]
    anchor_map = utils.make_labels_contiguous_vectorized(anchor_map)

    output = {
        "merged_map": anchor_map.unsqueeze(0),
        "merged_color_map": merged_color_map
    }
    return output


def fill_bev_map(xy, labels, bev_dim):
    """
    Fills bev map with xy and labels using max counts per patch

    Inputs:
        xy: torch.Tensor(N, 2)
        labels: torch.Tensor(N)
        bev_dim: tuple(H, W)
    Outputs:
        bev_map: torch.Tensor(H, W)
    """
    loc1d = xy[:, 0] * bev_dim[0] + xy[:, 1]
    loc1d_indices = torch.sort(torch.unique(loc1d))[0]

    if labels.size(0) == 0:
        return torch.zeros((1, bev_dim[0], bev_dim[1]), dtype=torch.long).to(xy.device)

    num_classes = labels.max() + 1  # Assuming classes are labeled from 0 to labels max
    num_indices = loc1d_indices.max().item() + 1  # The number of unique indices

    bev_labels_th = utils.most_frequent_per_index(labels, loc1d, num_classes)

    xy_sc = torch.stack([loc1d_indices // bev_dim[1],
                        loc1d_indices % bev_dim[1]], dim=1)
    bev_map = torch.zeros(
        (1, bev_dim[0], bev_dim[1]), dtype=torch.long).to(xy.device)
    bev_map[:, xy_sc[:, 1], xy_sc[:, 0]] = bev_labels_th + 1  # [H, W]

    return bev_map  # [H, W]


def calculate_iou(label1, label2):
    """
    Helper function to calculate the IoU between two labels.
    """
    intersection = torch.sum((label1 == label2) * (label1 > 0))
    union = torch.sum((label1 > 0) + (label2 > 0))
    if union == 0:
        return 0.0
    return intersection.float() / union.float()


def filter_ground_plane(xyz, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    """Return mask to remove ground plane points"""
    if type(xyz) == torch.Tensor:
        xyz = xyz.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)

    num_points = xyz.shape[0]
    inliers_mask = np.ones(num_points, dtype=bool)
    inliers_mask[inliers] = False

    # inlier_cloud = xyz[inliers_mask]
    # vis.numpy_to_pcd(inlier_cloud, "output.pcd")

    # inlier_cloud = pcd.select_by_index(inliers)
    # # save
    # o3d.io.write_point_cloud("output.pcd", inlier_cloud)
    # import pdb
    # pdb.set_trace()
    return inliers_mask


def compute_instance_cluster_iou(instance_array, cluster_array):
    """
    Computes the Intersection over Union (IoU) between each instance ID and each cluster ID.

    Args:
        instance_array (np.ndarray): A 2D array of shape (N, 1) containing instance IDs.
        cluster_array (np.ndarray): A 2D array of shape (N, 1) containing cluster IDs.

    Returns:
        iou_matrix (np.ndarray): A 2D array of shape (num_instances, num_clusters) containing IoU values.
        instance_ids (np.ndarray): 1D array of unique instance IDs corresponding to rows of iou_matrix.
        cluster_ids (np.ndarray): 1D array of unique cluster IDs corresponding to columns of iou_matrix.
    """
    # Reshape the arrays to 1D
    instance_array_flat = instance_array.reshape(-1)
    cluster_array_flat = cluster_array.reshape(-1)

    # Get unique instance and cluster IDs
    instance_ids = np.unique(instance_array_flat)
    cluster_ids = np.unique(cluster_array_flat)

    # Create mappings from IDs to indices
    instance_id_to_index = {id_: idx for idx, id_ in enumerate(instance_ids)}
    cluster_id_to_index = {id_: idx for idx, id_ in enumerate(cluster_ids)}

    # Map IDs to indices
    instance_indices = np.vectorize(
        instance_id_to_index.get)(instance_array_flat)
    cluster_indices = np.vectorize(cluster_id_to_index.get)(cluster_array_flat)

    # Compute the contingency table (intersection counts)
    data = np.ones_like(instance_indices)
    contingency = coo_matrix(
        (data, (instance_indices, cluster_indices)),
        shape=(len(instance_ids), len(cluster_ids))
    ).toarray()

    # Compute the total pixels for each instance and cluster
    instance_pixel_counts = np.bincount(
        instance_indices, minlength=len(instance_ids))
    cluster_pixel_counts = np.bincount(
        cluster_indices, minlength=len(cluster_ids))

    # Compute the union for each instance-cluster pair
    union = instance_pixel_counts[:, np.newaxis] + \
        cluster_pixel_counts[np.newaxis, :] - contingency

    # Compute IoU, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = contingency / union
        iou_matrix[np.isnan(iou_matrix)] = 0  # Replace NaN with 0

    return iou_matrix, instance_ids, cluster_ids


def cluster_xyz_labels(xyz, xyz_pt, dynamic_label):
    """
    Given the xyz point cloud and dynamic labels for each point, cluster points into groups using DBSCAN
    with varying densities and merge the labels if the IoU exceeds a predefined threshold.

    Inputs:
        xyz - (torch.Tensor) [B, N, 2] xyz points, where H and W are the pixel locations in the image for each point
        xyz_pt - (torch.Tensor) [B, N, 2] pixels corresponding to the xyz points when projected onto dynamic label image
        dynamic_label - (torch.Tensor) [B, 2, H, W] dynamic labels for each point, where the first channel
                        denotes the instance ID and the second channel denotes the class ID

    Returns:
        output_labels - (torch.Tensor) [B, N, 2] filtered labels for each point after clustering
    """
    B, N, _ = xyz.shape
    _, _, H, W = dynamic_label.shape
    output_labels = torch.zeros((B, N, 2), dtype=torch.long).to(xyz.device)
    xyz_pt = xyz_pt.long()
    dynamic_label = dynamic_label.permute(0, 2, 3, 1).long()

    label_mask = torch.zeros((B, H, W), dtype=torch.bool).to(xyz.device)
    label_mask[torch.arange(B).unsqueeze(
        1), xyz_pt[:, :, 1], xyz_pt[:, :, 0]] = True
    sparse_dynamic_label = torch.zeros_like(dynamic_label)
    sparse_dynamic_label[label_mask] = dynamic_label[label_mask]

    # """ BEGIN DEBUG IMAGE """
    # # Create image from xyz_pt and dynamic_label
    # test_img = torch.zeros((B, H, W, 3)).float()
    # random_output_colormap = vis.generate_cmap(
    #     sparse_dynamic_label.max().item() + 1)[:, :3]
    # random_output_colormap[0] = [0.7, 0.7, 0.7]  # Set background to black
    # test_img = torch.from_numpy(
    #     random_output_colormap[sparse_dynamic_label[:, :, :, 0]]).float()
    # # Save image as cv2
    # cv2.imwrite("test.png", test_img[0].cpu().numpy()*255)
    # """ END DEBUG IMAGE """

    predefined_threshold = 0.2  # Define the IoU threshold
    eps_list = [0.1, 0.2, 0.3]  # List of varying densities for DBSCAN
    min_samples_list = [5, 3, 5]

    for b in range(B):
        # Extract data for the current batch
        xyz_b_full = xyz[b]          # [N, 2]
        xyz_pt_b_full = xyz_pt[b]    # [N, 2]
        dynamic_label_b_full = dynamic_label[b]  # [H, W, 2]

        # Get original labels for all points
        xyz_pt_np_full_np = xyz_pt_b_full.cpu().numpy()
        dynamic_label_b_full_np = dynamic_label_b_full.cpu().numpy()
        all_instance_ids = dynamic_label_b_full_np[xyz_pt_np_full_np[:,
                                                                     1], xyz_pt_np_full_np[:, 0], 0]
        all_class_ids = dynamic_label_b_full_np[xyz_pt_np_full_np[:,
                                                                  1], xyz_pt_np_full_np[:, 0], 1]
        original_labels = np.stack((all_instance_ids, all_class_ids), axis=1)
        N_total = xyz_pt_np_full_np.shape[0]

        # Filter ground plane points
        fg_mask = filter_ground_plane(xyz_b_full)
        xyz_b = xyz_b_full[fg_mask]
        xyz_pt_b = xyz_pt_b_full[fg_mask]
        fg_instance_ids = all_instance_ids[fg_mask]
        fg_class_ids = all_class_ids[fg_mask]
        fg_original_labels = np.stack((fg_instance_ids, fg_class_ids), axis=1)

        # Convert tensors to numpy arrays
        xyz_np = xyz_b.cpu().numpy()
        xyz_pt_np = xyz_pt_b.cpu().numpy()
        N_points = xyz_np.shape[0]

        # Initialize assigned labels (0 indicates unlabeled)
        output_labels_b = np.zeros((N_points, 2), dtype=int)

        # Track IoU matrices for all eps and min_samples combinations
        iou_results = []
        cluster_labels_list = []
        if xyz_np.shape[0] == 0:
            # Pass original dynamic instance and class labels if no points are present
            output_labels[b] = torch.zeros((N_total, 2), dtype=torch.long).to(xyz.device)
            continue

        for eps, min_samples in zip(eps_list, min_samples_list):
            # Perform DBSCAN clustering with the current density
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(xyz_np)  # [N_points]
            if np.min(cluster_labels) == -1:
                cluster_labels += 1  # Shift labels to avoid -1 (noise)

            cluster_labels_list.append(cluster_labels)

            # Compute IoU matrix
            iou_matrix, instance_ids_unique, cluster_ids_unique = compute_instance_cluster_iou(
                fg_instance_ids, cluster_labels
            )

            iou_results.append(
                (iou_matrix, instance_ids_unique, cluster_ids_unique))

        # Assign labels based on the best IoU across all configurations
        all_fg_instance_ids_unique = np.unique(fg_instance_ids)

        max_instance_info = {
            instance: {"iou": 0, "iou_idx": -1, "cluster_idx": -1} for instance in all_fg_instance_ids_unique}
        matched_instance_ids = set()

        # Ensemble all dbscan clusters and map each instance to the best cluster
        for idx, (iou_matrix, instance_ids_unique, cluster_ids_unique) in enumerate(iou_results):
            for cluster_id in cluster_ids_unique:
                if cluster_id == 0:
                    continue

                cluster_mask = cluster_labels_list[idx] == cluster_id

                max_iou = np.max(iou_matrix[:, cluster_id])
                max_instance_id = instance_ids_unique[np.argmax(
                    iou_matrix[:, cluster_id])]

                best_iou_so_far = max_instance_info[max_instance_id]["iou"]

                if max_iou >= predefined_threshold and max_iou > best_iou_so_far:
                    max_instance_info[max_instance_id]["iou"] = max_iou
                    max_instance_info[max_instance_id]["iou_idx"] = idx
                    max_instance_info[max_instance_id]["cluster_idx"] = cluster_id

        for instance, info in max_instance_info.items():
            if info["cluster_idx"] != -1:
                cluster_mask = cluster_labels_list[info["iou_idx"]
                                                   ] == info["cluster_idx"]
                output_labels_b[cluster_mask, 0] = instance
                output_labels_b[cluster_mask,
                                1] = fg_original_labels[cluster_mask, 1]

        matched_instance_ids = set(output_labels_b[:, 0])
        # For unmatched instances in all points, retain the original instance and class ID
        unmatched_instance_ids = set(
            all_fg_instance_ids_unique) - matched_instance_ids

        output_labels_b_full = np.zeros((N_total, 2), dtype=int)
        output_labels_b_full[fg_mask] = output_labels_b

        for unmatched_instance_id in unmatched_instance_ids:
            unmatched_mask = (output_labels_b_full[:, 0] == 0) # [N_total]
            matching_instance_id_mask = (all_instance_ids == unmatched_instance_id) # [N_total]
            full_mask = unmatched_mask & matching_instance_id_mask

            output_labels_b_full[full_mask, 0] = unmatched_instance_id
            output_labels_b_full[full_mask, 1] = all_class_ids[full_mask]

        # for point_idx in range(N_total):
        #     original_instance_id = original_labels[point_idx, 0]
        #     original_class_id = original_labels[point_idx, 1]

        #     if original_instance_id not in matched_instance_ids and output_labels_b_full[point_idx, 0] == 0:
        #         output_labels_b_full[point_idx, 0] = original_instance_id
        #         output_labels_b_full[point_idx, 1] = original_class_id

        # Convert to torch tensor and assign to output_labels
        output_labels_b_torch = torch.from_numpy(
            output_labels_b_full).to(xyz.device)
        output_labels[b] = output_labels_b_torch

    # Verify that the unique labels match
    output_unique_labels = output_labels[:, :, 0].unique()
    sparse_unique_labels = sparse_dynamic_label[:, :, :, 0].unique()

    # Some segmentation masks will not have any instances if
    # all points it is associated with are assigned to a different mask
    # try:
    #     assert torch.equal(torch.sort(output_unique_labels)[0], torch.sort(sparse_unique_labels)[0]), \
    #     "Unmatched instances between original and output labels"
    # except AssertionError as e:
    #     print(e)
    #     import pdb; pdb.set_trace()

    return output_labels

def inflate_borders_batchwise(tensor, iterations=1):
    """
    Inflates the borders for each channel in a BxCxHxW tensor using dilation.
    
    Args:
        tensor (torch.Tensor): A BxCxHxW tensor where:
            - The first channel is the instance ID (-1 is unlabeled)
            - The second channel is the class ID (0 is unlabeled)
        iterations (int): Number of times to inflate the borders (default is 1).
    
    Returns:
        torch.Tensor: The tensor with inflated borders for each channel.
    """
    # Create a 3x3 kernel for dilation
    kernel = torch.ones((1, 1, 3, 3), device=tensor.device)

    # Function to perform dilation for a single channel in the batch
    def dilate_channel(channel, unlabeled_value):
        for _ in range(iterations):
            # Pad the channel to handle borders
            padded_channel = F.pad(channel, (1, 1, 1, 1), value=unlabeled_value)
            
            # Apply dilation using max pooling
            dilated = F.max_pool2d(padded_channel, kernel_size=3, stride=1, padding=0)
            
            # Keep original unlabeled values where dilation didn't overwrite
            # dilated = torch.where(channel == unlabeled_value, dilated, channel)
            
            # Update the channel with the dilated version
            channel = dilated
        return channel

    # Separate the batch of instance and class channels
    instance_channel = tensor[:, 0]  # Shape: BxHxW
    class_channel = tensor[:, 1]     # Shape: BxHxW

    # Dilate both instance and class channels for the entire batch
    dilated_instance = dilate_channel(instance_channel, unlabeled_value=0)
    dilated_class = dilate_channel(class_channel, unlabeled_value=0)

    # Stack the two channels back together for each batch
    dilated_tensor = torch.stack([dilated_instance, dilated_class], dim=1)

    return dilated_tensor

def compute_sam_map_single(inputs):
    """This function builds a bird's eye view labeled map from SAM mask labels
    It iteratively assigns the most probable label between sequential images
    """
    # mv_horizon, sam_horizon, depth_horizon, pose_horizon, pose_ref_idx, calib_dict, bev_params, img_ds, outpath_dict = inputs
    sam_dynamic_horizon, sam_static_horizon, pc_horizon, depth_horizon, pose_horizon, pose_ref_idx, calib_dict, bev_params, img_ds, outpath_dict = inputs
    ds_img_shape = (calib_dict['img_H'], calib_dict['img_W'])
    horizon = len(sam_static_horizon)

    # 1 Load movability, sam, and depth masks
    # mv_horizon = load_mvlabel_horizon(mv_horizon, ds_img_shape)
    sam_static_horizon = load_sam_horizon(
        sam_static_horizon, ds_img_shape, "static")  # [1, H, W]
    sam_dynamic_horizon = load_sam_horizon(
        sam_dynamic_horizon, ds_img_shape, "dynamic")  # [2, H, W]
    depth_horizon = load_depth_horizon(depth_horizon, ds_img_shape)
    pc_horizon = load_pc_horizon(pc_horizon)

    sam_static_horizon = torch.from_numpy(
        np.stack(sam_static_horizon)).float()  # [B, 1, H, W]
    sam_dynamic_horizon = torch.from_numpy(
        np.stack(sam_dynamic_horizon)).float()  # [B, 2, H, W]
    padded_sam_dynamic_horizon = inflate_borders_batchwise(
        sam_dynamic_horizon, iterations=12
    )
    depth_horizon = torch.from_numpy(
        np.stack(depth_horizon)).float().unsqueeze(1)  # [B, 1, H, W])
    pc_horizon = torch.from_numpy(np.stack(pc_horizon)).float()  # [B, N, 3]
    depth_horizon_mask = (depth_horizon[:, 0] > 0) & (
        depth_horizon[:, 0] < 12.8)  # [B, H, W]

    # 2 Compute p2p with poses
    pose_horizon = ch.convert_poses_to_tf(pose_horizon)
    pose_horizon = geo.transform_poses(pose_horizon, ref_idx=pose_ref_idx)

    pose_horizon = torch.from_numpy(pose_horizon).float()
    p2p_horizon = load_p2p(calib_dict, pose_horizon,
                           ds=1)  # already downsampled
    bev_dims = [int(dim) for dim in bev_params['map_size']]

    """ BEGIN Compute Dynamic Map Label """
    dynamic_label = sam_dynamic_horizon[pose_ref_idx]  # [2, H, W]
    xyz = pc_horizon[pose_ref_idx]
    # Filter out points that are out of range in bev
    xyz_mask = (xyz > bev_params['min_bound']) & (xyz < bev_params['max_bound'])
    xyz_mask = xyz_mask.all(dim=1).bool()
    ground_plane_mask = torch.from_numpy(filter_ground_plane(xyz)).bool()
    min_ground_mask = (xyz[:, 2] > -0.5)  # Filter out points below robot
    ground_mask = xyz_mask & ground_plane_mask & min_ground_mask

    # Filter out ground plane and out of range points
    xyz_ground = xyz.clone()[ground_mask]
    xyz = xyz[xyz_mask] # [N, 3]

    xyz_pts, xyz_mask = pixels_to_depth(
        xyz, calib_dict, ds_img_shape[0], ds_img_shape[1], return_keys=[
            'pc_pts', 'pc_mask']
    )  # [N, 2], [N]
    xyz = xyz[xyz_mask]  # [N, 3]
    xyz_pts = torch.from_numpy(xyz_pts).float()  # [N, 2]
    cluster_dynamic_label = cluster_xyz_labels(
        xyz.unsqueeze(0), xyz_pts.unsqueeze(0), dynamic_label.unsqueeze(0)).squeeze(0)  # [N, 2]
    xy = points2voxels((xyz.unsqueeze(0), bev_params)).squeeze(0)  # [N, 2]
    xy_occ = points2voxels((xyz_ground.unsqueeze(0), bev_params)).squeeze(0)  # [N, 2]
    xy_mask = (xy[:, 0] >= 0) & (xy[:, 0] < bev_dims[1]) & (
        xy[:, 1] >= 0) & (xy[:, 1] < bev_dims[0])
    xy_occ_mask = (xy_occ[:, 0] >= 0) & (xy_occ[:, 0] < bev_dims[1]) & (
        xy_occ[:, 1] >= 0) & (xy_occ[:, 1] < bev_dims[0])
    xy_masked = xy[xy_mask]
    xy_occ_masked = xy_occ[xy_occ_mask]
    cluster_dynamic_label_masked = cluster_dynamic_label[xy_mask]

    dynamic_map = torch.zeros((3, bev_dims[0], bev_dims[1])).long()
    dynamic_map[0, xy_masked[:, 1], xy_masked[:, 0]
                ] = cluster_dynamic_label_masked[:, 0]  # Save instance ids
    dynamic_map[1, xy_masked[:, 1], xy_masked[:, 0]
                ] = cluster_dynamic_label_masked[:, 1]  # Save class ids
    dynamic_map[2, xy_occ_masked[:, 1], xy_occ_masked[:, 0]] += 1
    cv2.imwrite("occupancy_test.png", dynamic_map[2].cpu().numpy()*255)
    # torch.save(xyz, f"xyz.pt")
    # torch.save(cluster_dynamic_label, f"dynamic_label.pt")
    """ END Compute Dynamic Map Label """

    """ BEGIN Compute Static Map Label """
    # 3 Compute Static Map Label
    xyz, xyz_mask = cam2world((depth_horizon, p2p_horizon, bev_params))
    # height_horizon_mask = (xyz[:, 2] > -2) & (xyz[:, 2] < 0.5)  # [B, H, W] # CODa
    height_horizon_mask = (xyz[:, 2] > -1.5) & (xyz[:, 2] < 1.0)  # [B, H, W] # CREStE
    B, _, H, W = xyz.shape
    xyz_flat = xyz.view(B, -1, H*W).permute(0, 2, 1)  # [B, HW, 3]
    xyz_mask_flat = xyz_mask.view(B, H*W)
    xy_horizon = points2voxels((xyz_flat, bev_params))

    # 4 Iteratively merge mask labels
    merged_color_map = None
    merged_map = torch.zeros((1, bev_dims[0], bev_dims[1])).long()

    # 4 Compute merged map label
    horizon_ids = np.array(
        [idx for idx in range(horizon) if idx != pose_ref_idx])
    horizon_ids = [pose_ref_idx] + horizon_ids.tolist()
    for idx, hidx in enumerate(horizon_ids):
        height_mask = height_horizon_mask[hidx].view(H*W)  # [HW]
        depth_mask = depth_horizon_mask[hidx].view(H*W)  # [HW]
        grid_mask = xyz_mask_flat[hidx]  # [HW]

        dynamic_label = padded_sam_dynamic_horizon[hidx].view(2, H*W)  # [HW]
        # 0 is unlabeled (static), 1 is labeled (dynamic)
        mv_mask = dynamic_label[0] == 0
        static_label = sam_static_horizon[hidx].view(H*W).long()  # [HW]
        xy = xy_horizon[hidx]  # [HW, 2]

        # 4b Apply masks (movability and in bev grid)
        mask = mv_mask & grid_mask & depth_mask & height_mask  # [HW]
        xy_masked = xy[mask]  # [N, 2]
        # [N] Labels start at 0 -> need to increment by 1
        label_masked = static_label[mask]

        # bev_map = torch.zeros((1, bev_dims[0], bev_dims[1])).long()
        # bev_map[0, xy_masked[:, 1], xy_masked[:, 0]] = label_masked + 1 # 0 unlabeled
        bev_map = fill_bev_map(xy_masked, label_masked, bev_dims)  # [1, H, W]
        bev_map = utils.make_labels_contiguous_vectorized(bev_map)

        # 4c Merge with previous map
        if idx == 0:
            num_labels = len(torch.unique(bev_map))
            merged_map = bev_map
            merged_color_map = vis.generate_cmap(num_labels, use_th=True)
            # Set background to transparent
            merged_color_map[0] = torch.zeros(4)
            merged_color_map[0, 3] = 1  # Set background to opaque
            # visualize_bev_map(merged_map, cmap=merged_color_map)
            assert len(torch.unique(merged_map)) == torch.max(
                merged_map)+1, "Labels are not contiguous"
        else:
            # TODO: LEverage view overlap amount to decide whether to merge maps
            # TODO: USe the ref pose as the first pose to merge with
            unmerged_map = torch.cat((merged_map, bev_map), dim=0)
            oppose_to_anchor = compute_label_mapping(unmerged_map, ignore=0)
            output = merge_maps(
                unmerged_map,
                merged_color_map,
                oppose_to_anchor,
                ignore=0
            )

            merged_map = output["merged_map"]
            merged_color_map = output["merged_color_map"]
    """ END Compute Static Map Label """
    dynamic_color_map = np.hstack(
        (np.array(SAM_DYNAMIC_COLOR_MAP) / 255.0, np.ones((len(SAM_DYNAMIC_COLOR_MAP), 1)))
    )  # [N, 4]
    dynamic_color_map = torch.from_numpy(dynamic_color_map).float()
    visualize_bev_map(dynamic_map[1:2],
                      cmap=dynamic_color_map, filepath="dynamic_test.png")
    visualize_bev_map(merged_map, cmap=merged_color_map,
                      filepath="static_test.png")
    
    # Save the merged map labels and rgb image
    visualize_bev_map(merged_map, cmap=merged_color_map,
                      filepath=outpath_dict['static_rgb'])
    visualize_bev_map(dynamic_map[1:2], cmap=dynamic_color_map,
                      filepath=outpath_dict['dynamic_rgb'])
    merged_map_np = merged_map.squeeze(0).cpu().numpy().astype(np.uint16)
    np.save(outpath_dict['static'], merged_map_np)
    # merged_map_np.tofile(outpath_dict['static'])
    dynamic_map_np = dynamic_map.cpu().numpy().astype(np.uint16)
    np.save(outpath_dict['dynamic'], dynamic_map_np)
    # dynamic_map_np.tofile(outpath_dict['dynamic'])
    print(f'Saved {outpath_dict["static"]} and {outpath_dict["dynamic"]}')

def extract_forward_windows(input_np, horizon):
    """
    Intakes NxK dimension input and returns a NxHxK forward windowed view of the input
    """
    # Create sliding window views of the poses array with window over the first dimension
    pad_input_np = np.pad(input_np, ((0, horizon-1), (0, 0)), mode='edge')
    sliding_windows = sliding_window_view(pad_input_np, window_shape=(
        horizon,), axis=0).transpose(0, 2, 1)  # [N, H, K]
    return sliding_windows


def extract_custom_windows(input_np, horizon, horizon_ref):
    """
    Intakes NxK dimension input and returns a NxHxK forward windowed view of the input
    """
    rear_horizon = horizon_ref
    front_horizon = horizon - horizon_ref - 1

    # Create sliding window views of the poses array with window over the first dimension
    pad_input_np = np.pad(
        input_np, ((rear_horizon, front_horizon), (0, 0)), mode='edge')
    sliding_windows = sliding_window_view(pad_input_np, window_shape=(
        horizon,), axis=0).transpose(0, 2, 1)  # [N, H, K]

    return sliding_windows

_NUM_RE = re.compile(r"\d+")


def _extract_seq_and_frame(path: Union[str, Path]) -> Tuple[int, int]:
    """
    Extracts the sequence **ID** (directory name just above the file) and
    the **frame number** (single integer in the file-stem) from the given path.

    Example
    -------
    >>> _extract_seq_and_frame(".../cam0/0/mask_505.npy")
    (0, 505)
    """
    p = Path(path)
    seq_str: str = p.parent.name                # last directory = sequence id
    seq_id: int = int(seq_str)                  # guaranteed numeric

    # stem is e.g. 'mask_505' or '505'
    nums = _NUM_RE.findall(p.stem)
    if not nums:
        raise ValueError(f"No frame number found in filename: {p}")
    if len(nums) > 1:
        raise ValueError(
            f"More than one integer found in filename, "
            f"ambiguous frame id extraction: {p}"
        )
    frame_id: int = int(nums[0])
    return seq_id, frame_id


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
def pad_discontinuous_frames(
    input_np: np.ndarray, match_np: np.ndarray
) -> List[Optional[str]]:
    """
    Aligns `input_np` files to the reference timeline encoded by `match_np`.

    Parameters
    ----------
    input_np : np.ndarray, shape (N, 1) or (N,)
        Paths containing *any* single integer inside the filename (frame id).
        Example row: `data/creste/.../cam0/0/mask_505.npy`
    match_np : np.ndarray, shape (M, 1) or (M,)
        Reference paths defining the desired *length* and *(seq, frame)* slots.
        Example row: `data/creste/.../cam0/0/505.npy`

    Returns
    -------
    List[Optional[str]]
        A list of length `len(match_np)` where each element is:
        * the matching path from `input_np` (same `(seq, frame)`), or
        * `None` when no match exists.
    """

    # --- normalise inputs to 1-D lists of strings -------------------------- #
    input_paths: List[str] = [str(x) for x in np.ravel(input_np)]
    match_paths: List[str] = [str(x) for x in np.ravel(match_np)]

    # --- build lookup table from (seq, frame) -> path ---------------------- #
    lookup: Dict[Tuple[int, int], str] = {}
    for p in input_paths:
        seq, frame = _extract_seq_and_frame(p)
        lookup[(seq, frame)] = p   # if duplicates, keep last one

    # --- construct padded list -------------------------------------------- #
    padded: List[Optional[str]] = []
    for ref in match_paths:
        seq, frame = _extract_seq_and_frame(ref)
        padded.append(lookup.get((seq, frame), None))

    return padded

def main(args):
    indir = args.indir
    outdir = args.outdir
    skip_factor = args.skip_factor
    camid = args.camid
    img_ds = args.img_ds
    horizon = args.horizon
    horizon_ref = args.horizon_ref
    skip_sequences = args.skip_sequences

    bev_params = {
        'min_bound': torch.Tensor([-12.8, -12.8, -2]).float().reshape(1, -1),
        'max_bound': torch.Tensor([12.8, 12.8, 1]).float().reshape(1, -1),
        'lidar2map': torch.Tensor([
            [0, -1, 0, 12.8],
            [-1, 0, 0, 12.8],
            [0, 0, -1, 0],  # Was 12.8
            [0, 0, 0, 1]
        ]).float(),
        'voxel_size': torch.Tensor([0.1, 0.1, 3]).float()
    }
    bev_params['map_size'] = (
        (bev_params['max_bound'] - bev_params['min_bound']) / bev_params['voxel_size']).squeeze().long()

    # 1 Build list of frames to process
    sensor_dict = [[CAMERA_DIR, CAMERA_SUBDIRS[0]]]
    input_frames_dict = get_frames_from_json(
        indir, sensor_dict, override_all=True, ds=skip_factor)

    unified_infos = np.empty((0, 2), dtype=int)
    for seq in input_frames_dict.keys():
        if seq in skip_sequences:
            continue
        for sensorid in input_frames_dict[seq].keys():
            seq_infos = np.ones_like(
                input_frames_dict[seq][sensorid], dtype=int) * seq
            frame_infos = np.array(
                [fn2frame(frame_path) for frame_path in input_frames_dict[seq][sensorid]], dtype=int
            )

            infos = np.stack((seq_infos, frame_infos), axis=1)
            unified_infos = np.vstack((unified_infos, infos))

    print("LOADED FRAMES FOR SEQUENCES\n", np.unique(unified_infos[:, 0]))

    # 2 Initialize inputs for compute movability mask
    # Output directories
    outpath_dir_dict = {
        "static": [join(outdir, "static"), "npy"],
        "dynamic": [join(outdir, "dynamic"), "npy"],
        "static_rgb": [join(outdir, "static_rgb"), "jpg"],
        "dynamic_rgb": [join(outdir, "dynamic_rgb"), "jpg"],
    }

    if img_ds == 1:
        sam_root_dir = join(indir, "sam2")
    else:
        sam_root_dir = join(indir, f'downsampled_{img_ds}', "sam2")

    static_label_dir = join(sam_root_dir, "static", CAMERA_DIR, camid)
    dynamic_label_dir = join(sam_root_dir, "dynamic", CAMERA_DIR, camid)
    depth_dir = join(indir, f'downsampled_{img_ds}', "depth_50_IDW_all")
    if not os.path.exists(depth_dir):
        depth_dir = join(indir, "depth_0_LAIDW_all")
    pc_dir = join(indir, POINTCLOUD_DIR, POINTCLOUD_SUBDIR[0])

    bev_params_list = []
    img_ds_list = []
    horizon_ref_list = []
    calib_dict_list = []
    pose_windows_list = []
    # mv_windows_list = []
    static_sam_windows_list = []
    dynamic_sam_windows_list = []
    pc_windows_list = []
    depth_windows_list = []
    for seq in np.unique(unified_infos[:, 0]):
        if seq in skip_sequences:
            continue

        # Load poses
        for subdir in POSES_SUBDIRS:
            pose_path = join(indir, POSES_DIR, subdir, f'{seq}.txt')
            if os.path.exists(pose_path):
                break
        assert os.path.exists(pose_path), f"Pose path {pose_path} does not exist"
        poses = np.loadtxt(pose_path, dtype=np.float64)

        dynamic_sam_horizon_paths = ch.get_dir_frame_info(
            join(dynamic_label_dir, str(seq)), ext='npy', short=False)
        dynamic_sam_horizon_paths = np.array(
            dynamic_sam_horizon_paths).reshape(-1, 1)
        assert len(dynamic_sam_horizon_paths) > 0, "No dynamic SAM labels found for sequence {}".format(seq)

        # Load SAM labels
        static_sam_horizon_paths = ch.get_dir_frame_info(
            join(static_label_dir, str(seq)), ext='npy', short=False)
        static_sam_horizon_paths = np.array(
            static_sam_horizon_paths).reshape(-1, 1)
        assert len(static_sam_horizon_paths) > 0, "No static SAM labels found for sequence {}".format(seq)

        dynamic_sam_horizon_paths = pad_discontinuous_frames(
            dynamic_sam_horizon_paths, static_sam_horizon_paths)
        dynamic_sam_horizon_paths = np.array(
            dynamic_sam_horizon_paths).reshape(-1, 1)

        # Extract the frame number using the last number in the filepath
        assert len(static_sam_horizon_paths) == len(
            dynamic_sam_horizon_paths), "SAM Label lengths do not match"

        # Extract all frames of in sam paths
        # import pdb; pdb.set_trace()
        # frames = np.array(
        #     [fn2frame(path[0]) for path in static_sam_horizon_paths], dtype=str)
        # import pdb; pdb.set_trace()

        # Load depth maps
        depth_horizon_paths = ch.get_dir_frame_info(
            join(depth_dir, str(seq), camid), ext='png', short=False)
        depth_horizon_paths = np.array(depth_horizon_paths).reshape(-1, 1)
        pc_horizon_paths = ch.get_dir_frame_info(
            join(pc_dir, str(seq)), ext='bin', short=False)
        # Filter pc_horizon_paths to match frames of interest
        frames_of_interest = unified_infos[unified_infos[:, 0] == seq][:, 1]
        pc_horizon_paths = np.array(
            [pc_path for pc_path in pc_horizon_paths if fn2frame(
                pc_path) in frames_of_interest]
        ).reshape(-1, 1)
        assert np.max(frames_of_interest) < len(poses), "Frame index exceeds pose length"
        poses = poses[frames_of_interest]  # [N, 8] 

        # # Load infos for each seq frame
        # info_paths = ch.get_dir_frame_info(join(infos_dir, str(seq)), ext='pkl', short=False)
        # frame_infos = select_frames_from_info(info_paths)

        # Load windows
        pose_windows = extract_custom_windows(
            poses, horizon, horizon_ref)  # [N, H, 8]
        # mv_windows = extract_custom_windows(
        #     mv_horizon_paths, horizon, horizon_ref)  # [N, H, 1]
        static_sam_windows = extract_custom_windows(
            static_sam_horizon_paths, horizon, horizon_ref)
        dynamic_sam_windows = extract_custom_windows(
            dynamic_sam_horizon_paths, horizon, horizon_ref)  # [N, H, 1]
        depth_windows = extract_custom_windows(
            depth_horizon_paths, horizon, horizon_ref)  # [N, H, 1]
        pc_windows = extract_custom_windows(
            pc_horizon_paths, horizon, horizon_ref)  # [N, H, 1]

        if len(pc_windows) != len(depth_windows) or len(pose_windows) != len(static_sam_windows):
            import pdb; pdb.set_trace()
        # Load calibrations
        calib_dict = {}
        calib_dict.update(ch.load_intrinsics(indir, seq, camid))
        calib_dict.update(ch.load_extrinsics(indir, seq, camid))
        calib_dict = ch.scale_calib(calib_dict, 1.0/img_ds)

        # Save to lists
        calib_dict_list.extend([calib_dict] * len(pose_windows))
        bev_params_list.extend([bev_params] * len(pose_windows))
        img_ds_list.extend([img_ds] * len(pose_windows))
        horizon_ref_list.extend([horizon_ref] * len(pose_windows))
        pose_windows_list.extend(pose_windows)
        # mv_windows_list.extend(mv_windows)
        static_sam_windows_list.extend(static_sam_windows)
        dynamic_sam_windows_list.extend(dynamic_sam_windows)
        depth_windows_list.extend(depth_windows)
        pc_windows_list.extend(pc_windows)

        # Make output directories
        for outpath_list in outpath_dir_dict.values():
            os.makedirs(join(outpath_list[0], str(seq)), exist_ok=True)

    outpath_dict_list = [
        {k: join(v[0], str(seq), f'{frame}.{v[1]}')
         for k, v in outpath_dir_dict.items()}
        for seq, frame in unified_infos if seq not in skip_sequences
    ]

    assert len(calib_dict_list) == len(pose_windows_list) == len(static_sam_windows_list) == len(dynamic_sam_windows_list) == len(pc_windows_list) == len(
        depth_windows_list) == len(outpath_dict_list) == len(bev_params_list), "Lengths do not match"

    task_args_list = [
        (dynamic_windows, static_windows, pc_windows, depth_windows, pose_windows,
         horizon_ref, calib_dict, bev_params, img_ds, outpath_dict)
        for dynamic_windows, static_windows, pc_windows, depth_windows, pose_windows, horizon_ref, calib_dict, bev_params, img_ds, outpath_dict in zip(
            dynamic_sam_windows_list, static_sam_windows_list, pc_windows_list, depth_windows_list, pose_windows_list, horizon_ref_list, calib_dict_list, bev_params_list, img_ds_list, outpath_dict_list
        )
    ]
    # import pdb; pdb.set_trace()
    # compute_sam_map_single(task_args_list[1000]) # 12 14830
    # compute_sam_map_single(task_args_list[33065]) # 13 9955 ped + car
    # compute_sam_map_single(task_args_list[17029])  # 7 7750 ped + car
    # compute_sam_map_single(task_args_list[47536])  # 20 7135 indoors peds
    # for i in range(0, len(task_args_list), 1000):
    #     compute_sam_map_single(task_args_list[i])

    # for i in range(11452, 11652, 1):
    #     compute_sam_map_single(task_args_list[i])

    # for i in range(0, len(task_args_list), 1):
    #     compute_sam_map_single(task_args_list[i])

    # Launch worker pool for process each frame with progress bar
    # with Pool(24) as pool:
    #     with tqdm(total=len(task_args_list)) as pbar:
    #         for _ in pool.imap(compute_sam_map_single, task_args_list):
    #             pbar.update()
    # import pdb; pdb.set_trace()
    # task_args_list = task_args_list[6723:]
    # Start parallel processing joblib with progress bar
    num_workers = 48  # Adjust this based on your system
    Parallel(n_jobs=num_workers, backend='loky', verbose=1)(delayed(compute_sam_map_single)(task_args) for task_args in tqdm(task_args_list))


if __name__ == '__main__':
    args = parse_args()
    main(args)