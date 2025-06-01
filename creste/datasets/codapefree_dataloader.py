"""
Dataloader for learning PE free dino features
"""

import os
import copy
from os.path import join
import yaml
import glob
import cv2
import pickle
import time
import re
from collections import OrderedDict
from tqdm import tqdm

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from creste.utils.projection import get_pixel2pts_transform, get_pts2pixel_transform, pixels_to_depth, points2voxels
from creste.utils.visualization import visualize_dino_feature, visualize_rgbd_3d, visualize_bev_label, visualize_elevation_3d_wrapper, visualize_bev_poses
from creste.utils.train_utils import ImageAugmentation, create_trapezoidal_fov_mask, RotateAndTranslate, median_filter_2d, expand_filter_2d, balanced_infos_resampling, DepthAugmentation
from creste.utils.utils import remap_and_sum_channels_torch, make_labels_contiguous_vectorized, drop_overlapping_horizons

from creste.datasets.coda_utils import *
import creste.datasets.coda_helpers as ch


class CodaPEFreeDataset(Dataset):
    def __init__(self,
                 cfg,
                 split="training",
                 camids=['cam0'],
                 use_global_pose=False,
                 do_augmentation=False,
                 dataset_idx=0,
                 **kwargs,
                 ):
        # Make copy of cfg to avoid changing original cfg
        self.cfg = copy.deepcopy(cfg)

        print("------ BEGIN CODaPEFreeDataset Init ------")

        """Merge kwargs overrides with cfg"""
        for key, value in kwargs.items():
            if key in self.cfg:
                print(f'Overriding {key} in cfg with {value}')
            print(f'Setting {key} to {value}')
            self.cfg[key] = value
        """End merge kwargs"""

        """Add all paramters in cfg to self"""
        for key, value in self.cfg.items():
            setattr(self, key, value)

        # Override split, camids, use_global_pose, do_augmentation
        self.split = split
        self.camids = camids
        self.use_global_pose = use_global_pose
        self.do_augmentation = do_augmentation
        self.dataset_idx = dataset_idx

        """BEGIN TASK CONFIG SETUP"""
        # Load from task configs
        task_cfgs = {}
        for task in self.task_cfgs:
            task_cfgs[task['name']] = task['kwargs']
        self.__dict__["task_cfgs"] = task_cfgs
        """END TASK CONFIG SETUP"""

        """BEGIN SSC specific parameters"""
        self.fsc_label_subdir, self.sam_label_subdir, self.soc_label_subdir, self.ssc_label_subdir, self.elevation_label_subdir = None, None, None, None, None
        if "distillation" in self.task_cfgs.keys():
            self.do_distillation = True
        else:
            self.do_distillation = False
        print("Setting up ssc configs")
        self.setup_ssc(cfg)
        """END SSC specific parameters"""
        print("------ END CODaPEFreeDataset Init ------")

        if 'fimg_pred' in self.fload_keys:
            assert os.path.isdir(
                self.fimg_pred_dir), "Feature prediction directory not provided"

        self.pc_dir = join(self.root_dir, POINTCLOUD_DIR, 'os1')
        if self.ds_rgb == 1:
            self.rgb_dir = join(self.root_dir, '2d_rect')
            self.depth_dir = join(self.root_dir, f'depth_0_LA_all')
            self.gt_depth_dir_list = [
                join(self.root_dir, f'downsampled_{self.ds_gt_depth}', f'depth_0_{self.infill_strat}_all'),
            ]
        else:
            self.rgb_dir = join(
                self.root_dir, f'downsampled_{self.ds_rgb}', '2d_rect')
            self.depth_dir = join(
                self.root_dir, f'downsampled_{self.ds_rgb}', 'depth_1_LA_all')
            self.gt_depth_dir_list = [
                join(self.root_dir,
                    f'downsampled_{self.ds_gt_depth}', 'depth_50_IDW_all'),
                join(
                    self.root_dir, f'downsampled_{self.ds_gt_depth}', 'depth_50_IDW_semanticsegmentation'),
                join(self.root_dir,
                    f'downsampled_{self.ds_gt_depth}', 'depth_50_IDW_object')
            ]

        self.immovable_dir = join(self.root_dir, '3d_comp_movability', 'os1')
        # self.immovable_label_dir = join(self.root_dir,
        #                                 f'downsampled_{self.ds_gt_depth}',
        #                                 f'{CAMERA_DIR}_movability',
        #                                 'mask'
        #                                 )
        self.immovable_label_dir = join(self.root_dir, 'sam2', 'dynamic', CAMERA_DIR, CAMERA_SUBDIRS[0])
        if not os.path.exists(self.immovable_label_dir):
            self.immovable_label_dir = join(self.root_dir, f'downsampled_{self.ds_rgb}', 'sam2', 'dynamic', CAMERA_DIR, CAMERA_SUBDIRS[0])

        self.pose_dir = join(self.root_dir, POSES_DIR, POSES_SUBDIRS[0])
        self.pose_dir = join(self.root_dir, POSES_DIR, POSES_SUBDIRS[1]) if not os.path.exists(self.pose_dir) else self.pose_dir
        assert os.path.exists(self.pose_dir), f"Pose directory not found at {self.pose_dir}"
        self.pose_dict = {}

        # seq_list = np.arange(0, 23).tolist()
        seq_list = ch.get_available_sequences(self.root_dir)
        for seq in seq_list:
            if seq in self.skip_sequences:
                continue
            pose_np = np.loadtxt(join(self.pose_dir, f'{seq}.txt'))
            # Transform to R and T matrix from quaternion and translation
            pose_th = torch.eye(4, dtype=torch.float32).repeat(
                len(pose_np), 1, 1)
            pose_th[:, :3, 3] = torch.from_numpy(pose_np[:, 1:4]).float()
            # [w, x, y, z] -> [x, y, z, w]
            quat = np.hstack([pose_np[:, 5:8], pose_np[:, 4:5]])
            pose_th[:, :3, :3] = torch.from_numpy(
                R.from_quat(quat).as_matrix()
            ).float()
            self.pose_dict[seq] = pose_th

        # infos and features only defined for camera zero
        # check if dino_dir exists as attribute of class
        assert hasattr(self, 'dino_dir') or hasattr(self, 'info_dir'), "Dino and info dir not found in config"
        if not hasattr(self, 'info_dir'):
            self.info_dir = self.dino_dir
        if not hasattr(self, 'dino_dir'):
            self.dino_dir = self.info_dir

        # Default to using depth unless specified
        if not hasattr(self, 'use_depth'):
            self.use_depth = True

        # if not self.do_distillation:
        #     # Don't need overlap if not distilling
        #     self.infos_dir = join(self.info_dir, "infos_nooverlap", "cam0")
        # else: # TODO: Check if this is correct for coda
        self.infos_dir = join(self.info_dir, "infos", "cam0")

        self.IMG_H = self.cfg['img_h'] // self.ds_rgb
        self.IMG_W = self.cfg['img_w'] // self.ds_rgb
        self.GT_DEPTH_H = self.cfg['img_h'] // self.ds_gt_depth
        self.GT_DEPTH_W = self.cfg['img_w'] // self.ds_gt_depth

        # Loads all samples defined in infos and feats files
        if self.split == "all":
            self.split = "full"
        self._load_split_files(self.split)

        # Setup data augmentations
        self.setup_augmentations()

    def setup_ssc(self, cfg):
        """
        Setup SSC and SOC specific parameters
        """
        self.BEV_H = int(self.cfg['map_size'][0] // self.cfg['voxel_size'][0])
        self.BEV_W = int(self.cfg['map_size'][1] // self.cfg['voxel_size'][1])
        self.frustrum_mask = create_trapezoidal_fov_mask(
            self.BEV_H,
            self.BEV_W,
            70, 70, 7, 200
        )

        # Loading subdirectories for each label
        if f'fimg_label' in self.fload_keys:
            self.gt_feats_dir = self.task_cfgs[DISTILLATION_LABEL_DIR]['subdir']
            assert os.path.exists(
                self.gt_feats_dir), f"GT features subdir not found at {self.gt_feats_dir}"
        if f'{FSC_LABEL_DIR}_label' in self.sload_keys:
            self.fsc_label_subdir = self.task_cfgs[FSC_LABEL_DIR]['subdir']
            assert os.path.exists(
                self.fsc_label_subdir), f"Feature label subdir not found at {self.fsc_label_subdir}"
        if f'{SAM_LABEL_DIR}_label' in self.sload_keys:
            self.sam_label_subdir = self.task_cfgs[SAM_LABEL_DIR]['subdir']
            assert os.path.exists(
                self.sam_label_subdir), f"SAM label subdir not found at {self.sam_label_subdir}"
        if f'{SAM_DYNAMIC_LABEL_DIR}_label' in self.sload_keys:
            self.sam_dynamic_label_subdir = self.task_cfgs[SAM_DYNAMIC_LABEL_DIR]['subdir']
            assert os.path.exists(
                self.sam_dynamic_label_subdir), f"SAM dynamic label subdir not found at {self.sam_dynamic_label_subdir}"
        if f'{SOC_LABEL_DIR}_label' in self.sload_keys:
            self.soc_label_subdir = join(self.root_dir, SOC_LABEL_DIR)
            assert os.path.exists(
                self.soc_label_subdir), f"SOC label subdir not found at {self.soc_label_subdir}"
        if f'{SSC_LABEL_DIR}_label' in self.sload_keys:
            self.ssc_label_subdir = join(self.root_dir, SSC_LABEL_DIR)
            assert os.path.exists(
                self.ssc_label_subdir), f"SSC label subdir not found at {self.ssc_label_subdir}"
        if f'{ELEVATION_LABEL_DIR}_label' in self.sload_keys:
            if f'{SAM_LABEL_DIR}_label' in self.sload_keys:
                # self.elevation_label_subdir = join(
                #     self.root_dir, ELEVATION_LABEL_DIR)
                self.elevation_label_subdir = join(
                    self.task_cfgs[ELEVATION_LABEL_DIR]['subdir']
                )
            elif f'{SSC_LABEL_DIR}_label' in self.sload_keys:
                print(f'Using SSC label subdir for elevation')
                self.elevation_label_subdir = join(
                    self.root_dir, f'{ELEVATION_LABEL_DIR}_ssc')
            assert os.path.exists(
                self.elevation_label_subdir), f'Elevation label subdir not found at {self.elevation_label_subdir}'
        if f'{COUNTERFACTUAL_LABEL_DIR}_label' in self.sload_keys:
            self.counterfactual_label_subdir = join(
                self.root_dir, COUNTERFACTUAL_LABEL_DIR, COUNTERFACTUAL_LABEL_SUBDIR[0])
            assert os.path.exists(
                self.counterfactual_label_subdir), f'Counterfactual label subdir not found at {self.counterfactual_label_subdir}'
        if TASK_TO_LABEL[TRAVERSE_LABEL_DIR] in self.sload_keys:
            self.traversability_label_subdir = join(
                self.root_dir, TRAVERSE_LABEL_DIR, TRAVERSE_LABEL_SUBDIR[0])
            assert os.path.exists(
                self.traversability_label_subdir), f'Traversability label subdir not found at {self.traversability_label_subdir}'

    def setup_augmentations(self):
        """
        Setup data augmentations for the images
        """
        self.image_augmentation = ImageAugmentation(
            **self.cfg['camera_augmentation'])
        self.pc_augmentation = RotateAndTranslate(
            self.cfg['pc_augmentation'],
            self.cfg['map_size'],
            self.cfg['voxel_size']
        )
        self.pc_augmentation.renew_transformation()

        self.depth_augmentation = DepthAugmentation(
            **self.cfg['depth_augmentation']
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to be used with the CODaDataset.
        Args:
            batch: List of dictionaries with keys 'image', 'point_cloud', 'calibration'
                'pose', 'label'
                where 'image' is a numpy image, 'point_cloud' is a numpy array, and
                'calibration' is a dictionary, 'pose' is a dictionary, 'label' is a numpy array
        Returns:
            A dictionary with keys 'image', 'point_cloud', 'calibration', 'pose',
            'label' where 'image' is a tensor stack of all images, 'point_cloud' is a list of numpy arrays
            (since point clouds can have varying lengths, they can't be stacked into a
            single tensor), 'calibration' is a list of calibration dictionaries, 'pose' is
            a list of pose dictionaries, 'label' is a tensor stack of all labels
        """
        samples = {}

        for key in batch[0].keys():
            if type(batch[0][key]) == torch.Tensor:
                samples[key] = torch.stack([d[key] for d in batch], axis=0)
            else:
                samples[key] = [d[key] for d in batch]

        return samples

    def _load_split_files(self, split):
        """Loads split file for a given task set
        Split files use the following naming convention:
            {task1}_{taskn}/{split}.txt
        """
        split_dir = self.datasets[self.dataset_idx]['split_dir']
        # Sort tasks before creating task string
        split_path = join(split_dir, f'{split}.txt')
        assert os.path.exists(split_path), f"Split file not found at {split_path}"
        
        # Uses split idx to load samples
        self.idx_to_sample = np.loadtxt(split_path).astype(int)
        min_deviation = self.cfg.get('min_deviation', 0)
        split_distances_path = join(split_dir, f'{split}_distances.txt')
        resample = self.cfg.get('resample_trajectories', False)        
        if os.path.exists(split_distances_path):
            samples = np.array([f'{sample[0]} {sample[1]}' for sample in self.idx_to_sample], dtype=str)
            distances = np.loadtxt(split_distances_path).astype(float)

            if split == 'training' and resample:
                samples, distances = balanced_infos_resampling(
                    samples, distances, num_bins=20
                )

            # Keep trajectories that satisfy min_distance
            samples = samples[distances >= min_deviation]
            distances = distances[distances >= min_deviation]

            # Update idx to sample dictionary
            self.idx_to_sample = np.array([sample.split(' ') for sample in samples], dtype=int)

        # Filter out sequences in skip sequence
        if len(self.skip_sequences) > 0:
            self.idx_to_sample = np.array(
                [sample for sample in self.idx_to_sample if sample[0] not in self.skip_sequences], dtype=int)

        # Create global idx to load samples
        if self.do_distillation and self.views > 1:
            global_idx_to_sample = []
            seq_subdirs = ch.get_sorted_subdirs(self.gt_feats_dir)

            for seq_subdir in seq_subdirs:
                seq = os.path.basename(seq_subdir)
                infos_files = ch.get_dir_frame_info(
                    seq_subdir, ext=self.task_cfgs[DISTILLATION_LABEL_DIR]['ext'], short=True
                )
                global_idx_to_sample.extend(infos_files)

            # Format for 'seq_frame' string to idnex
            self.local_idx_to_sample = np.array(
                [f'{seq}_{frame}' for seq, frame in self.idx_to_sample], dtype=str)
            self.global_idx_to_sample = np.array(
                [s.replace(' ', '_') for s in global_idx_to_sample], dtype=str)
        
        print(f'Loaded {len(self.idx_to_sample)} samples from {split_path}')

    def get_num_labels(self, task):
        return len(self.get_class_labels(task))

    def get_class_labels(self, task):
        if SSC_LABEL_DIR in self.task_cfgs:
            ssc_remap = self.task_cfgs[SSC_LABEL_DIR]['remap_labels']
        if SOC_LABEL_DIR in self.task_cfgs:
            soc_remap = self.task_cfgs[SOC_LABEL_DIR]['remap_labels']

        if task == SSC_LABEL_DIR:
            return SEM_LABEL_REMAP_CLASS_NAMES if ssc_remap else SEM_LABEL_CLASS_NAMES
        elif task == SOC_LABEL_DIR:
            return OBJ_LABEL_REMAP_CLASS_NAMES if soc_remap else OBJ_LABEL_NAMES
        elif task == SAM_DYNAMIC_LABEL_DIR:
            return SAM_DYNAMIC_LABEL_NAMES
        elif task == SAM_LABEL_DIR:
            return []
        elif task == ELEVATION_LABEL_DIR:
            return []
        else:
            raise NotImplementedError(f"Task {task} not found in class labels")

    def get_bev_params(self):
        output = {
            "map_size": self.cfg['map_size'],
            "voxel_size": self.cfg['voxel_size'],
            "map_range": self.cfg['map_range']
        }
        return output

    def __len__(self):
        return len(self.idx_to_sample)

    def _transform_poses(self, poses):
        """
        Computes transformation of poses from global frame to current frame
        Inputs:
            poses - list of poses in global frame  (4x4 numpy array)
        Outpus:
            rel_poses - list of poses in current frame (4x4 numpy array)
        """

        # 1 Compute relative poses
        T_ref_global = poses[0]
        pose_horizon = torch.empty((0, 4, 4), dtype=torch.float32)
        for pose in poses:
            T_global = pose
            T_rel = torch.linalg.inv(T_ref_global) @ T_global
            pose_horizon = torch.cat(
                (pose_horizon, T_rel.unsqueeze(0)), axis=0)

        return pose_horizon

    def add_load_key(self, key, num_frames):
        """
        Returns correct data structure for given load key to output dict
        """
        image_channels = 3
        if self.use_depth:
            image_channels = 4

        if key == "image":
            return torch.zeros(num_frames, image_channels, self.IMG_H, self.IMG_W)
        elif key in ['p2p', 'p2p_in']:
            return torch.zeros(num_frames, 4, 4)
        elif key == "depth_label":
            return torch.zeros(num_frames, self.GT_DEPTH_H, self.GT_DEPTH_W)
        elif key in ["fimg_label", "fimg_pred"]:  # Dino pe or pefree feature label
            return torch.zeros(num_frames, self.fimg_shape[2], self.fimg_shape[0], self.fimg_shape[1])
        elif key == "pose":
            return torch.zeros(num_frames, 4, 4)
        elif key == "sequence":
            return torch.zeros(num_frames).long()
        elif key == "frame":
            return torch.zeros(num_frames).long()
        elif key == "point_cloud":
            return torch.zeros(num_frames, POINTS_PER_SCAN, 3)
        elif key == "immovable_label":
            return torch.zeros(num_frames, POINTS_PER_SCAN, 1).bool()
        elif key == "immovable_depth_label":
            return torch.zeros(num_frames, self.IMG_H, self.IMG_W).bool()
        elif key == "fov_mask":
            return torch.zeros(self.BEV_H, self.BEV_W).bool()
        # Below is for scene wise data
        elif key in LABEL_TO_TASK.keys():
            task = LABEL_TO_TASK[key]
            if task == TRAVERSE_LABEL_DIR:
                return torch.zeros(num_frames, 2, 2)
            return torch.zeros(num_frames, self.task_cfgs[task]['num_classes'], self.BEV_H, self.BEV_W)
        else:
            raise NotImplementedError(f"Key {key} not found in load_keys")

    def load_frame_data(self, overlap_infos, key):
        """
        Load prebev data for the given key
        """
        if key == "sequence":
            return int(overlap_infos['id'].split('_')[0])
        elif key == "frame":
            return int(overlap_infos['id'].split('_')[1])
        elif key == "image":
            # TODO: Investigate effect of augmentation of same scene
            return self._load_rgbd(overlap_infos, keep_aug_mask=False)
        elif key == "depth_label":
            return self._load_depth_label(overlap_infos)
        elif key == "sem_label":
            return self._load_sem_label(overlap_infos)
        elif key == "fimg_label":
            return self._load_fimg_label(overlap_infos)
        elif key == "fimg_pred":
            return self._load_fimg_pred(overlap_infos)
        elif key == "pose":
            return self._load_pose_from_info(overlap_infos)
        elif key == "point_cloud":
            return self._load_point_cloud(overlap_infos)
        elif key == "immovable_label":
            return self._load_immovable_label(overlap_infos)
        elif key == "immovable_depth_label":
            return self._load_immovable_depth_label(overlap_infos)
        # Scene wise data
        elif key in LABEL_TO_TASK.keys():
            task = LABEL_TO_TASK[key]
            return self._load_scene_data(overlap_infos, task)
        else:
            raise NotImplementedError(f"Key {key} not found in load_keys")

    def __getitem__(self, idx):
        seq, frame = self.idx_to_sample[idx]

        # 1 Loads calibrations, poses, and overlapping global indexes
        infos = self._load_infos(seq, frame)

        num_frames = self.views
        overlap_indices = [infos['id']]
        if self.do_distillation and self.views > 1:
            # TODO: Debug this after code changes from ssc
            # 2 Randomly select global indexes from overlapping views
            aux_overlap_indices = self._select_overlap_indices(infos)

            # 3 Load rgbd image inputs, ground truth depth, and p2p for each view
            overlap_indices += aux_overlap_indices
            num_frames += len(self.camids)
        elif self.views > 0:  # views > 1
            # Load sequential overlapping indices
            overlap_indices += self._select_sequential_indices(infos)
        
        # 3a Setup output dictionary
        load_keys = self.fload_keys + self.sload_keys
        output = {}
        for load_key in load_keys:
            # num_frames = self.views if load_key != "image" else len(self.camids)
            num_frames = self.views
            output[load_key] = self.add_load_key(load_key, num_frames)
        try:
            # 3b Load frame wise data
            for idx, frame_id in enumerate(overlap_indices):
                seq, frame = frame_id.split('_')
               
                overlap_infos = self._load_infos(seq, frame)
                for key in self.fload_keys:
                    frame_data = self.load_frame_data(overlap_infos, key)
                    if not torch.is_tensor(frame_data) or (frame_data.shape[-2:] == output[key][idx].shape[-2:] and frame_data.shape[0] != output[key].shape[0]):
                        output[key][idx] = frame_data
                    elif frame_data.shape[0] == output[key].shape[0]:
                        output[key] = frame_data
                    else:
                        import pdb; pdb.set_trace() 
                        raise ValueError(f"Shape mismatch for key {key}")

            # 4 Convert poses to anchor view and calculate p2p
            if "pose" in self.fload_keys and not self.use_global_pose:
                output["pose"] = self._transform_poses(output["pose"])

            # 5 Load scene wise data
            for skey in self.sload_keys:
                # Special case since p2p is dependent on pose
                if skey == "p2p":
                    output.update(self._load_p2p(infos, output["pose"]))
                elif skey == "fov_mask":
                    output[skey] = self._load_fov_mask(output["pose"])
                else:
                    seq, frame = infos['id'].split('_')
                    overlap_infos = self._load_infos(seq, frame)
                    sout = self.load_frame_data(overlap_infos, skey)
                    output[skey] = sout
        except Exception as e:
            import pdb; pdb.set_trace() 
            print(f"Error loading data for {infos['id']}: {e} key {key}")
            raise e

        return output

    def _load_scene_data(self, infos, task):
        seq, frame = infos['id'].split('_')

        """Loads BEV Level Scene data"""
        label = None
        label_size = (self.BEV_H, self.BEV_W,
                      self.task_cfgs[task]['num_classes'])
        ext = self.task_cfgs[task].get('ext', '')
        if task == SAM_LABEL_DIR:
            label_path = join(self.sam_label_subdir, seq, f'{frame}.{ext}')
            kernel_size = self.task_cfgs[task].get('kernel_size', 5)
            label = self._load_sam(
                label_path, label_size, kernel_size=kernel_size)
        elif task == SAM_DYNAMIC_LABEL_DIR:
            label_path = join(self.sam_dynamic_label_subdir,
                              seq, f'{frame}.{ext}')
            kernel_size = self.task_cfgs[task].get('kernel_size', 5)
            label = self._load_sam(
                label_path, label_size, kernel_size=kernel_size)
        elif task == FSC_LABEL_DIR:
            label_path = join(self.fsc_label_subdir, seq, f'{frame}.bin')
            label = self._load_fsc(label_path, label_size)
        elif task == SOC_LABEL_DIR:
            label_path = join(self.soc_label_subdir, seq, f'{frame}.bin')
            remap_labels = self.task_cfgs[task]['remap_labels']
            label = self._load_soc(label_path, label_size, remap_labels)
        elif task == SSC_LABEL_DIR:
            label_path = join(self.ssc_label_subdir, seq, f'{frame}.bin')
            remap_labels = self.task_cfgs[task]['remap_labels']
            label = self._load_ssc(label_path, label_size, remap_labels)
        elif task == ELEVATION_LABEL_DIR:
            label_path = join(self.elevation_label_subdir, seq, f'{frame}.bin')
            label = self._load_elevation(label_path, label_size)
        elif task == TRAVERSE_LABEL_DIR:
            step_size = self.task_cfgs[task].get('step_size', 1)
            label = self._load_traverse(
                infos, self.task_cfgs[task]['num_views'], (self.BEV_H, self.BEV_W), step=step_size)
        elif task == COUNTERFACTUAL_LABEL_DIR:
            label_path = join(self.counterfactual_label_subdir, seq, f'{frame}.{ext}')
            label = self._load_counterfactuals(label_path)
        else:
            raise NotImplementedError(f"Task {task} not implemented")

        return label

    def _load_counterfactuals(self, label_path):
        if not os.path.exists(label_path):
            return None
        with open(label_path, 'rb') as f:
            counterfactuals = pickle.load(f)
        
        # Load counterfactuals in bev coordiante frame
        return counterfactuals
        
    def _load_traverse(self, infos, num_views, bev_size, step):
        # Load pose for each view to relative to first pose
        # pose_ids = [infos['id']]+self._select_sequential_indices(infos, views=num_views)
        seq, frame = infos['id'].split('_')
        seq, frame = int(seq), int(frame)

        frame_ids = frame + np.arange(0, int(num_views*step), step)
        pose_ids = [f'{seq}_{i}' for i in frame_ids]
        poses = [self._load_pose(*pose_id.split('_')) for pose_id in pose_ids]
        lidar_poses = self._transform_poses(poses)

        # Extract 2x2 rot and 2x1 translation matrix
        voxel_size = torch.tensor(self.cfg['voxel_size']).float()
        bev_lidar_poses = torch.eye(3, 3)
        bev_lidar_poses = bev_lidar_poses.repeat(num_views, 1, 1)
        bev_lidar_poses[:, :2, :2] = lidar_poses[:, :2, :2]
        bev_lidar_poses[:, :2, 2] = lidar_poses[:, :2, 3] / voxel_size

        # Transform poses to BEV space
        T_lidar_to_bev = torch.tensor([
            [-1, 0, bev_size[1]//2],
            [0, -1, bev_size[0]//2],
            [0, 0, 1]
        ], dtype=torch.float32)
        bev_poses = torch.matmul(T_lidar_to_bev, bev_lidar_poses)

        # Transform bev poses (m) to grid poses (m)
        grid_bev_poses = bev_poses.clone()
        grid_bev_poses[:, :2, 2] = bev_poses[:, :2, 2]

        # Clip to grid size and fov mask
        min_val = torch.tensor([0, 0]).float()
        max_val = torch.tensor(bev_size).float()
        grid_bev_poses[:, :2, 2] = torch.clamp(
            grid_bev_poses[:, :2, 2], min_val, max_val)
        # import pdb; pdb.set_trace()
        return grid_bev_poses

    def _load_elevation(self, label_path, label_size):
        # 1 Load elevation tensor
        dtype = np.float64 if 'elevation_ssc' in label_path else np.float32
        elevation_tensor = torch.from_numpy(np.fromfile(
            label_path, dtype=dtype
        )).float().reshape(label_size)
        elevation_tensor = elevation_tensor.permute(2, 0, 1)  # [C, H, W]

        return elevation_tensor

    def _load_sam(self, label_path, label_size, kernel_size):
        if label_path.endswith('.bin'):
            label = np.fromfile(label_path, dtype=np.uint16).reshape(label_size)
        elif label_path.endswith('.npy'):
            label = np.load(label_path)
            if label.shape[:2]!=label_size[:2]: # Check if [H,W,C] or [C,H,W] format
                label = np.moveaxis(label, 0, -1) 
            label = label.reshape(label_size)
        else:
            raise NotImplementedError(f"Label format not found at {label_path}")
        label = torch.from_numpy(label).long().permute(2, 0, 1)  # [C, H, W]

        # 1 Apply kernel size to label
        if 'static' in label_path:
            label = median_filter_2d(label.unsqueeze(0), kernel_size).squeeze(0)  # [C, H, W]
            # 2 Make new labels contiguous
            label = make_labels_contiguous_vectorized(label)
        elif 'dynamic' in label_path:
            # instance id, class id, and occupancy map 
            label = expand_filter_2d(label.unsqueeze(0), kernel_size).squeeze(0) # [C, H, W]

        return label.long()

    def _load_fsc(self, label_path, label_size):
        label = np.fromfile(label_path, dtype=np.float32).reshape(label_size)
        label = torch.from_numpy(label).float().permute(
            2, 0, 1)  # [H, W, F] -> [F, H, W]
        return label

    def _load_ssc(self, label_path, label_size, remap_labels):
        """Return ssc labels show number of counts of each class per voxel"""
        # 1 Load semantic label tensor
        sem_tensor = torch.from_numpy(np.fromfile(
            label_path, dtype=int
        )).reshape(label_size)  # [H W C]

        # 2 Remap labels and change label dimension to reduced remapped label size
        if remap_labels:
            sem_tensor = remap_and_sum_channels_torch(
                sem_tensor,
                SEM_LABEL_REMAP
            )
        sem_tensor = sem_tensor.permute(2, 0, 1)  # [C, H, W]

        return sem_tensor

    def _load_soc(self, label_path, label_size, remap_labels):
        # 1 Load object label tensor
        soc_tensor = torch.from_numpy(
            np.fromfile(label_path,
                        dtype=np.uint16, offset=0,
                        count=np.prod(label_size)).astype(np.float32)
        ).reshape(label_size)

        # 2 Remap labels
        if remap_labels:
            soc_tensor = remap_and_sum_channels_torch(
                soc_tensor,
                OBJ_LABEL_REMAP
            )
        soc_tensor = soc_tensor.permute(2, 0, 1)  # [C, H, W]

        return soc_tensor

    def _load_fov_mask(self, pose_horizon):
        """Loads a BEV mask for the current frame"""
        accum_fov_mask = torch.zeros(
            (self.BEV_H, self.BEV_W), dtype=torch.bool)

        # 1 Accumulate fov mask
        for pose_idx, pose in enumerate(pose_horizon):
            # 3 Transform frustrum mask to orientation stable reference frame
            mask = self.frustrum_mask.clone().unsqueeze(-1).long()

            # 4 Transform frustrum mask to current frame
            RT = self.pc_augmentation.compute_transformation_fromSE3(pose)
            mask, _ = self.pc_augmentation.transform_map(mask, R_init=RT)

            # 4 Join masks together
            accum_fov_mask = accum_fov_mask | mask.squeeze()
            if pose_idx == 0:  # Only mask from current fov
                break
        return accum_fov_mask.bool()

    def _load_fimg_label(self, infos):
        """Load Dino PE Feature label"""
        seq, frame = infos['id'].split('_')

        # 1 Load PE dino feature
        feature_label = torch.zeros(
            0, self.fimg_shape[2], self.fimg_shape[0], self.fimg_shape[1])
        for idx, camid in enumerate(self.camids):
            if idx >= self.views:
                break
            feat_path = join(self.gt_feats_dir, seq, f'{frame}.npy')
            feat_th = torch.from_numpy(np.load(feat_path)).float()
            feature_label = torch.cat(
                [feature_label, feat_th.unsqueeze(0)], dim=0)

        return feature_label

    def _load_fimg_pred(self, infos):
        """Loads PEFREE dino feature label """
        seq, frame = infos['id'].split('_')

        # 1 Load feature prediction
        feat_path = join(self.fimg_pred_dir, seq, f'{frame}.pt')
        assert os.path.exists(
            feat_path), f"Feature prediction not found at {feat_path}"
        feat = torch.load(feat_path).float()
        return feat

    def _load_immovable_depth_label(self, infos):
        """
        Loads a point cloud and its immovable mask, backprojects point cloud to depth pixel space.
        Then creates an immovable mask for the depth image

        TODO: Move this to preprocessing step for faster data loading
        """
        seq, frame = infos['id'].split('_')

        immovable_depth_mask = torch.ones(
            1, self.IMG_H, self.IMG_W).bool()
        mask_path = join(self.immovable_label_dir, seq, f'mask_{frame}.npy')
        if not os.path.exists(mask_path):
            return immovable_depth_mask
        mask_np = np.load(mask_path)
        immovable_depth_mask[0, mask_np>0] = 0 # Set background class to immovable

        # for camid in self.camids:
        #     mask_path = join(self.immovable_label_dir,
        #                      camid, seq, f'{frame}.png')
        #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #     mask = torch.from_numpy(mask).bool()
        #     immovable_depth_mask = torch.cat(
        #         [immovable_depth_mask, mask.unsqueeze(0)], dim=0)

        return immovable_depth_mask

    def _load_immovable_label(self, infos):
        seq, frame = infos['id'].split('_')
        # True if point is static and False if point is non-static
        # Load movability of points
        mv_path = join(self.immovable_dir, seq, f'{frame}.bin')
        immovable = np.fromfile(mv_path, dtype=bool).reshape(-1, 1)

        return torch.from_numpy(immovable).bool()

    def _load_point_cloud(self, infos):
        seq, frame = infos['id'].split('_')

        # 1 Load point cloud
        fname = frame2fn(POINTCLOUD_DIR, "os1", seq, frame, "bin")
        pc_path = join(self.pc_dir, seq, fname)
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(POINTS_PER_SCAN,
                                                            FEATURES_PER_POINT)[:, :3]  # Only extract xyz
        pc = torch.from_numpy(pc).float()

        return pc

    def _load_pose(self, seq, frame):
        seq, frame = int(seq), int(frame)
        frame = min(frame, len(self.pose_dict[seq])-1)
        # assert frame < len(self.pose_dict[seq]), f"Frame {frame} not found in sequence {seq}"
        pose = self.pose_dict[seq][frame]
        return pose

    def _load_pose_from_info(self, infos):
        seq, frame = infos['id'].split('_')
        pose = self._load_pose(seq, frame)
        return pose
    #     pose = torch.eye(4, dtype=torch.float32)
    #     pose[:3, :] = torch.from_numpy(
    #         infos['calib']['T_lidar_to_world']).float()[:3, :]
    #     return pose

    def _load_p2p(self, infos, poses=None):
        calib_dict_out = copy.deepcopy(infos['calib'])
        calib_dict_in = copy.deepcopy(infos['calib'])

        # Scale projection matrices for backprojection
        calib_dict_out['K'][:2, :] = calib_dict_out['K'][:2,
                                                         :] / self.ds_gt_depth
        calib_dict_out['P'][:2, :] = calib_dict_out['P'][:2,
                                                         :] / self.ds_gt_depth

        # Transform to orientation stable reference frame
        p2p = torch.tensor(
            get_pixel2pts_transform(calib_dict_out)
        ).float().unsqueeze(0)

        # Scale projection matrices for input debugging
        calib_dict_in['K'][:2, :] = calib_dict_in['K'][:2, :] / self.ds_rgb
        calib_dict_in['P'][:2, :] = calib_dict_in['P'][:2, :] / self.ds_rgb

        p2p_in = torch.tensor(
            get_pixel2pts_transform(calib_dict_in)
        ).float().unsqueeze(0)
        
        # Get pts to pixel transform
        pt2pix = torch.tensor(
            get_pts2pixel_transform(calib_dict_out)
        ).float().unsqueeze(0)

        pt2pix_in = torch.tensor(
            get_pts2pixel_transform(calib_dict_in)
        ).float().unsqueeze(0)

        if poses is not None:
            p2p = torch.matmul(poses, p2p)
            p2p_in = torch.matmul(poses, p2p_in)
            pt2pix = torch.matmul(pt2pix, poses)  # TODO: Debug this line
            pt2pix_in = torch.matmul(pt2pix_in, poses)

        return {"p2p": p2p, "pt2pix": pt2pix, "p2p_in": p2p_in, "pt2pix_in": pt2pix_in}

    def _load_rgbd(self, infos, keep_aug_mask):
        seq, frame = infos['id'].split('_')

        C = 3
        if self.use_depth:
            C = 4

        # 1 Load rgb image
        rgbd_th = torch.zeros(
            (0, C, self.IMG_H, self.IMG_W), dtype=torch.float32)
        for camid in self.camids:
            rgb_path = join(self.rgb_dir, camid, seq,
                            f'2d_rect_{camid}_{seq}_{frame}.png')
            if not os.path.exists(rgb_path):
                rgb_path = rgb_path.replace('png', 'jpg')
            rgb = cv2.cvtColor(cv2.imread(
                rgb_path, -1).astype(np.uint8), cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            if self.do_augmentation:
                rgb = self.image_augmentation(rgb, keep_aug=keep_aug_mask)

            if self.use_depth:
                # 2 Load depth image
                depth_path = join(self.depth_dir, seq, camid, f'{frame}.png')
                depth = cv2.imread(depth_path, -1).astype(np.float32)
                depth = torch.from_numpy(depth).float().unsqueeze(0)

                if self.do_augmentation:
                    depth = self.depth_augmentation(depth)
                # cv2.imwrite("test2.png", cv2.normalize(depth[0].numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                rgbd = torch.cat([rgb, depth], dim=0)
            else:
                rgbd = rgb

            rgbd_th = torch.cat([rgbd_th, rgbd.unsqueeze(0)], dim=0)

        return rgbd_th

    def _load_depth_label(self, infos):
        seq, frame = infos['id'].split('_')

        # 1 Load ground truth depth image
        depth_label = torch.zeros(0, self.GT_DEPTH_H, self.GT_DEPTH_W)
        for idx, camid in enumerate(self.camids): 
            if idx >= self.views:   # Only load as many depth labels as the number of views
                break

            for gt_depth_dir in self.gt_depth_dir_list:
                depth_path = join(gt_depth_dir, seq, camid, f'{frame}.png')
                if os.path.exists(depth_path):
                    break
            try:
                depth = cv2.imread(depth_path, -1).astype(np.float32)
            except AttributeError:
                print(f"Error loading depth image at {depth_path}")
            depth_th = torch.from_numpy(depth).float()
            depth_label = torch.cat(
                [depth_label, depth_th.unsqueeze(0)], dim=0)

        return depth_label

    def _load_feature_label(self, infos):
        seq, frame = infos['id'].split('_')

        # 1 Load PE dino feature
        feature_label = torch.zeros(
            0, self.fimg_shape[2], self.fimg_shape[0], self.fimg_shape[1])
        for camid in self.camids:
            feat_path = join(self.gt_feats_dir, seq, f'{frame}.npy')
            feat_th = torch.from_numpy(np.load(feat_path)).float()
            feature_label = torch.cat(
                [feature_label, feat_th.unsqueeze(0)], dim=0)

        return feature_label

    def _is_valid_frame(self, seq, frame):
        # Checks if ssc and elevation labels exist for frame
        valid = True
        if getattr(self, 'ssc_label_subdir', None) is not None:
            valid &= os.path.exists(
                join(self.ssc_label_subdir, seq, f'{frame}.bin'))
        if getattr(self, 'elevation_label_subdir', None) is not None:
            valid &= os.path.exists(
                join(self.elevation_label_subdir, seq, f'{frame}.bin'))
        if getattr(self, 'infos_dir', None) is not None:
            valid &= os.path.exists(join(self.infos_dir, seq, f'{frame}.pkl'))
        if getattr(self, 'soc_label_subdir', None) is not None:
            valid &= os.path.exists(
                join(self.soc_label_subdir, seq, f'{frame}.bin'))

        return valid

    def _select_sequential_indices(self, infos, views=-1, ds=5):
        """
        Select sequential indices for a given sequence. If frame dne, fill in 
        prior valid frame index. 
        """
        seq, frame = infos['id'].split('_')
        views = self.views if views == -1 else views

        # 1 Select future sequential frames (currently samples every 5 frames)
        nframe = frame
        selected_indices = []
        for i in range(1, views):
            qframe = int(frame) + i*ds

            if self._is_valid_frame(seq, qframe):
                nframe = qframe

            selected_indices.append(f'{seq}_{nframe}')
            print(f'Added seq {seq} frame {nframe}')
        return selected_indices

    def _select_overlap_indices(self, infos):
        """
        Randomly select overlapping indices for a given sequence
        """
        # overlap_indices = infos['overlap'].nonzero()[0]
        """TODO: Uncomment this once overlaps are done processing"""
        overlap_global_ids = infos['overlap_ids']
        overlap_pct = infos['overlap_ratio']
        overlap_local_ids = self.global_idx_to_sample[overlap_global_ids]

        # Remove ovelap indices that are no in info list or do not meet thresholds
        mask = np.in1d(overlap_local_ids, self.local_idx_to_sample)

        """ TODO: Uncomment this once overlaps are done processing"""
        mask = mask & \
            (overlap_pct > self.cfg['overlap_thresholds'][0]) & \
            (overlap_pct < self.cfg['overlap_thresholds'][1])
        overlap_pct = overlap_pct[mask]

        overlap_local_ids = overlap_local_ids[mask]

        need_replace = len(overlap_local_ids) < self.views
        # assert len(overlap_local_ids) >= self.views, \
        # f'Not enough overlapping views to select from for {infos["id"]}'
        if len(overlap_local_ids) == 0:  # Return Self
            return [infos['id']] * self.views
        selected_indices = np.random.choice(
            overlap_local_ids, self.views, replace=need_replace).tolist()

        # Randomly select V views
        return selected_indices

    def _load_infos(self, seq, frame):
        """
        Load infos file for a given index
        """
        info_path = join(self.infos_dir, str(seq), f'{frame}.pkl')
        assert os.path.exists(info_path), f"Info file not found at {info_path}"
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        return infos


if __name__ == '__main__':
    print("------   CODaPEFree Test   -----")
    # CODa
    # cfg_file = "./configs/dataset/traversability/coda_ds2_sam2elevtraverse_horizon50.yaml"
    # CREStE monocular
    # cfg_file = "./configs/dataset/traversability/creste_sam2elevtraverse_horizon_rgbonly.yaml"
    # CREStE stereo
    # cfg_file = "./configs/dataset/distillation/creste_pefree_dinov2_stereo.yaml"
    # CRESTE mono lidar
    cfg_file = "./configs/dataset/traversability/creste_sam2elevtraverse_horizon.yaml"
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    if cfg.get('action_horizon', None) is not None:
        cfg['action_horizon'] = 50
        # cfg['datasets'][0]['split_dir'] = 'data/creste_rlang/splits/3d_sam_3d_sam_dynamic_elevation_traversability_counterfactuals_hausdorff1m_horizon50_curvature' 
        cfg['datasets'][0]['split_dir'] = 'data/creste/splits/3d_sam_3d_sam_dynamic_elevation_traversability_counterfactuals_hausdorff0m_horizon50_curvature' 

    # 0 Initialize tracker instance and GroundedSAM
    train_dataset = CodaPEFreeDataset(
        cfg=cfg,
        split="full",
        views=1,  # Analagous to timestamp in this context
        # skip_sequences=[8, 14, 15, 22],
        # skip_sequences=[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22],
        skip_sequences=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        fload_keys=['sequence', 'frame', 'image', 'depth_label', 'pose'],
        # sload_keys=['p2p', 'fov_mask', "3d_sam_label", "3d_sam_dynamic_label", "elevation_label",
        #             f'{TASK_TO_LABEL[TRAVERSE_LABEL_DIR]}',  "3d_soc_label", "3d_ssc_label"],
        sload_keys=['p2p', 'fov_mask', "3d_sam_label", "3d_sam_dynamic_label", "elevation_label",
                    f'{TASK_TO_LABEL[TRAVERSE_LABEL_DIR]}'],
        # sload_keys = ['p2p', 'fov_mask', '3d_ssc_label', '3d_soc_label', 'elevation_label',  'counterfactuals_label', 'traversability_label'],
        # sload_keys=['p2p', '3d_sam_label', 'fov_mask'],
        # sload_keys=['p2p', '3d_sam_label', '3d_ssc_label', 'elevation_label', "fov_mask"],
        # sload_keys=['p2p', '3d_sam_label', '3d_ssc_label', '3d_soc_label', 'elevation_label', "fov_mask"],
        # sload_keys=['p2p', '3d_fsc_label', "3d_ssc_label", "fov_mask", "elevation_label"],
        # sload_keys=['p2p', "3d_sam_label", "fov_mask", "elevation_label"],
        task_cfgs=[
            {
                'name': SAM_LABEL_DIR,
                'kwargs': {
                    # 'subdir': "postprocess_rlang/build_map_outputs/sam2/static",
                    "subdir": "data/creste_rlang/sam2_map/static",
                    'num_classes': 1,
                    'kernel_size': 5,
                    'ext': "npy",
                }
            },
            {
                'name': SAM_DYNAMIC_LABEL_DIR,
                'kwargs': {
                    # 'subdir': "postprocess_rlang/build_map_outputs/sam2/dynamic",
                    "subdir": "data/creste_rlang/sam2_map/dynamic",
                    'num_classes': 3,
                    'kernel_size': 5,
                    'ext': "npy",
                }
            },
            {
                'name': ELEVATION_LABEL_DIR,
                "kwargs": {
                    "subdir": "data/creste_rlang/sam2_map/geometric/elevation/labels",
                    "num_classes": 2,
                    "ext": "bin"
                }
            },
            {
                'name': TRAVERSE_LABEL_DIR,
                'kwargs': {
                    # 'subdir': 'postprocess_rlang/build_map_outputs/traversability',
                    "subdir": "data/creste_rlang/poses/dense",
                    'num_views': 50,
                    'num_classes': 0,  # dummy variable
                    'step_size': 1,
                    'ext': "txt"
                }
            },
            # {
            #     'name': SOC_LABEL_DIR,
            #     "kwargs": {
            #         "remap_labels": True,
            #         "num_classes": 60, # Note remap=True will change the actual number of classes
            #         "ext": "bin"
            #     }
            # },
            # {
            #     'name': SSC_LABEL_DIR,
            #     "kwargs": {
            #         "remap_labels": True,
            #         "num_classes": 25,  # Note remap=True will change the actual number of classes
            #         "ext": "bin"
            #     }
            # },
            # {
            #     'name': COUNTERFACTUAL_LABEL_DIR,
            #     'kwargs': {
            #         'num_classes': 11,
            #         'ext': 'pkl'
            #     }
            # }
            # {
            #     'name': OCCUPANCY_LABEL_DIR,
            #     'kwargs': {
            #         'modality': '3d_comp',
            #         'sensor_name': 'os1',
            #         "num_classes": 1,
            #         "ext": "bin"
            #     }
            # }
        ],
        # camids=['cam0', 'cam1'],
        camids=['cam0'],
        do_augmentation=True
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=0
        # 6 workers is the sweet spot for B=3, V=2
    )

    total_time = 0
    start_time = time.time()
    num_samples = len(train_dataset)
    with tqdm(total=num_samples) as pbar:
        for i, batch in enumerate(dataloader):
            total_time += time.time() - start_time

            # Visualize dino features
            # print("Batch: ", i)
            # print(f'Sequence: {batch["sequence"]}, Frame: {batch["frame"]}')
            # # Reduce to [B*T*S, 4 ,H, W]
            # Visualize bev feature
            fov_mask = batch['fov_mask'][0].numpy()
            print("seq ", batch['sequence'][0], "frame ", batch['frame'][0].item())
            # if batch['frame'][0].item()==9360:
            if batch['sequence'][0].item()<19:
                continue
            if batch['frame'][0].item() < 4830:
                continue
            # Uncomment to save pickle file for compilation
            inference_dict = {
                "rgbd": batch['image'][0:1],
                "p2p": batch['p2p'][0:1],
                "elevation_label": batch['elevation_label'][0:1],
                # "3d_ssc_label": batch['3d_ssc_label'][0:1],
                "3d_sam_label": batch['3d_sam_label'][0],
                "3d_sam_dynamic_label": batch['3d_sam_dynamic_label'][0],
                "fov_mask": batch['fov_mask'][0:1],
                "depth_label": batch['depth_label'][0:1],
                "p2p_in": batch['p2p_in'][0:1],
            }
            # if batch['sequence'][0].item() != 6:
            #     continue
            torch.set_printoptions(threshold=5, sci_mode=False)
            with open("./runtime/data_dict_creste_19_4830.pkl", "wb") as f:
                pickle.dump(inference_dict, f)
                import pdb; pdb.set_trace()
            scene_image = np.empty((0, 256, 3), dtype=np.uint8)
            sam_image, sam_dynamic_image, fsc_image, ssc_image, elevation_image = None, None, None, None, None
            if '3d_fsc_label' in batch.keys():
                features = torch.stack(
                    [batch['3d_fsc_label'], batch['3d_fsc_label']], axis=0)
                fsc_image = visualize_bev_label(FSC_LABEL_DIR, features)
                H, W, _ = fsc_image.shape
                scene_image = np.concatenate(
                    [scene_image, fsc_image[:H//2, :W//2]], axis=0)
            # if '3d_sam_label' in batch.keys():
            #     features = torch.stack(
            #         [batch['3d_sam_label'], batch['3d_sam_label']], axis=0).squeeze(2)
            #     sam_image = visualize_bev_label(SAM_LABEL_DIR, features)
            #     H, W, _ = sam_image.shape  # 2 1 256 256
            #     sam_image = sam_image[:H, :W//2]
            #     sam_image[~fov_mask] = [0, 0, 0]

            #     scene_image = np.concatenate(
            #         [scene_image, sam_image[:H//2, :]], axis=0)
            # if '3d_sam_dynamic_label' in batch.keys():
            #     sam_dynamic_label = batch['3d_sam_dynamic_label'][0, 1:2]

            #     features = torch.stack(
            #         [sam_dynamic_label, sam_dynamic_label], axis=0)
            #     sam_dynamic_image = visualize_bev_label(
            #         SAM_DYNAMIC_LABEL_DIR, features
            #     )
            #     H, W, _ = sam_dynamic_image.shape
            #     sam_dynamic_image = sam_dynamic_image[:H, :W//2]
            #     sam_dynamic_image[~fov_mask] = [0, 0, 0]

            #     scene_image = np.concatenate(
            #         [scene_image, sam_dynamic_image[:H//2, :]], axis=0)
            # if '3d_ssc_label' in batch.keys():
            #     features = torch.stack(
            #         [batch['3d_ssc_label'], batch['3d_ssc_label']], axis=0)
            #     prob = features / \
            #         (torch.sum(features, dim=2, keepdim=True) + 1e-6)
            #     mode = torch.argmax(prob, dim=2)
            #     ssc_image = visualize_bev_label(SSC_LABEL_DIR, mode)
            #     H, W, _ = ssc_image.shape
            #     ssc_image = ssc_image[:H, :W//2]
            #     ssc_image[~fov_mask] = [0, 0, 0]
            #     scene_image = np.concatenate(
            #         [scene_image, ssc_image[:H//2, :]], axis=0)
            # if '3d_soc_label' in batch.keys():
            #     mode = torch.argmax(batch['3d_soc_label'], dim=1)
            #     mode = torch.stack([mode, mode], axis=0)
            #     soc_image = visualize_bev_label(
            #         SOC_LABEL_DIR, mode, remap_labels=True)
            #     H, W, _ = soc_image.shape
            #     soc_image = soc_image[:H, :W//2]
            #     soc_image[~fov_mask] = [0, 0, 0]

            #     scene_image = np.concatenate(
            #         [scene_image, soc_image[:H//2, :]], axis=0)
            if 'elevation_label' in batch.keys():
                # features = torch.stack([batch['elevation_label'], batch['elevation_label']], axis=0).squeeze()
                features = batch['elevation_label'].squeeze(1)
                features = features[:, 0, :, :]
                elevation_image = visualize_elevation_3d_wrapper(
                    features, features)
                H, W, _ = elevation_image.shape
                scene_image = np.concatenate(
                    [scene_image, elevation_image[H//2:, :W//2]], axis=0)

            # scene_image[~fov_mask] = 0
            if scene_image.shape[0] > 0:
                cv2.imwrite("bev_label.png", scene_image)
                seq, frame = batch['sequence'][0].item(), batch['frame'][0].item()
                save_path = f"./bev/{seq}/{frame}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, scene_image)

            if TASK_TO_LABEL[TRAVERSE_LABEL_DIR] in batch.keys() and sam_image is not None:
                visualize_bev_poses(
                    batch[TASK_TO_LABEL[TRAVERSE_LABEL_DIR]], img=sam_image, batch_idx=0)
                seq, frame = batch['sequence'][0].item(
                ), batch['frame'][0].item()

                # save_path = f"./traverse/{seq}/{frame}.jpg"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # cv2.imwrite(save_path, sam_image)
                cv2.imwrite("poses.png", sam_image)

            B = len(batch['sequence'])
            for b in range(B):
                bgr_img = batch['image'][b, 0, :3].permute(1, 2, 0).numpy()*255
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite("test.png", rgb_img)

            # B, V, C, H, W = batch['image'].shape
            # sf, ef = 0, 5
            # sf = 0
            # for ef in range(1, V):
            #     batch_img =  batch['image'].view(B*V, C, H, W)[sf:ef]
            #     batch_p2p_in = batch['p2p_in'].view(B*V, 4, 4)[sf:ef]
            #     batch_img = torch.stack([batch['image'][0][sf], batch['image'][0][ef]], axis=0)
            #     batch_p2p_in = torch.stack([batch['p2p_in'][0][sf], batch['p2p_in'][0][ef]], axis=0)
            #     visualize_rgbd_3d(batch_img, batch_p2p_in, num_scans=V, num_cams=1, filepath="test3d.png", z_max=1)

            #     if 'fimg_label' in batch.keys():
            #         vis_img_batch = torch.stack([batch['image'][0][0], batch['image'][0][ef]], axis=0)
            #         vis_feat_batch = torch.stack([batch['fimg_label'][0][0], batch['fimg_label'][0][ef]], axis=0)
            #         visualize_dino_feature(vis_img_batch, vis_feat_batch)

            # if i > num_samples:
            #     break
            start_time = time.time()
            pbar.update(1)

    print(f"Average time per sample: {total_time/num_samples}")
