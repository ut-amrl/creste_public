import os
from os.path import join

# Torch Imports
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from creste.utils.train_utils import ImageAugmentation

# Standard Imports
from PIL import Image
import cv2

import numpy as np
import json
import yaml

# Custom Imports
from creste.datasets.coda_utils import *

rng = np.random.default_rng(seed=42)

class CODatasetDepth(Dataset):
    """
    A custom PyTorch Dataset class to load images, point cloud binaries, and calibration files from a directory.
    Assumes the directory structure as provided in the screenshot.
    
    """  
    def __init__(
            self, 
            cfg,
            split="training",
            annos_type="Depth",
            skip_sequences=[],
            horizon=50,
            do_transforms=True,
            do_augmentation=True,
            inverse_depth=False,
            overfit=False
        ):
        """
        Args:
            root_dir (string): Directory with all the subdirectories and data.
        """
        self.cfg = cfg
        self.root_dir = cfg['root_dir']
        self.split = split
        self.annos_type = annos_type
        self.skip_sequences = skip_sequences
        self.horizon = horizon
        self.overfit = overfit
        self.ds_rgb = cfg['ds_rgb']
        self.ds_gt_depth = cfg['ds_gt_depth']

        self.depth_label_dir = join(
            self.root_dir, 
            f'downsampled_8',
            f'{DEPTH_DIR}_{self.horizon}_{self.cfg["infill_strat"]}_all'
        )

        # Augmentation Paramters
        self.do_transforms = do_transforms
        self.do_augmentation = do_augmentation
        self.inverse_depth = inverse_depth

        self.frames_list = self._load_frames()

        # Dataloader transformers
        if self.do_transforms:
            self.IMG_H = self.cfg['img_h'] // self.ds_rgb
            self.IMG_W = self.cfg['img_w'] // self.ds_rgb
            self.GT_DEPTH_H = self.cfg['img_h'] // self.ds_gt_depth
            self.GT_DEPTH_W = self.cfg['img_w'] // self.ds_gt_depth
        else:
            self.IMG_H = self.cfg['img_h']
            self.IMG_W = self.cfg['img_w']

        assert split in ["training", "validation", "testing", "all"], \
            f"Split {split} not recognized"

        # Load dataset
        self._load_data_paths(self.frames_list)

    def create_split_file(self, output_file):
        assert self.split != "all", "Cannot create split file for all splits"

        with open(output_file, 'w') as f:
            for frame_info in self.frames_list:
                seq, frame = frame_info
                f.write(f'{seq} {frame}\n')
            
        print("Split file created at {}".format(output_file))

    def get_num_labels(self):
        return max(SEM_LABEL_REMAP)+1 if self.remap_labels else len(SEM_LABEL_REMAP)

    def _get_frameinfo_from_filepath(self, filepath):
        """
        Return corresponding sequence and frame for input
        """
        seq, frame = os.path.splitext(os.path.basename(filepath))[0].split("_")[-2:]
        return seq, frame

    def _load_frames(self):
        """
        Reads train, validation, test split for the dataset if exists. Otherwise, reads all frames
        and builds train, val, test splits from scratch.
        
        Returns:
            frames_dict (list): List of sequence names and frame numbers to load
        """
        frames_list = []

        splits = ["training", "validation", "testing"] if self.split=="all" else [self.split]
        for split in splits:
            split_path = join(self.depth_label_dir, f'{split}.txt')
            
            if not os.path.exists(split_path):
                print(f'No {split} split file found at {split_path}, building from scratch')
                frames_list = self._build_split_frames(splits)

            #0 Load split file if exists
            print(f'Loading {split} split from {split_path}')
            frames_np = np.loadtxt(split_path, dtype=str)
            frames_list.extend(frames_np.tolist())

        # Sort frames by sequence and frame number if requesting to load all
        if self.split=="all":
            frames_list.sort(key=lambda x: (int(x[0]), int(x[1])))
        
        if self.overfit:
            frames_list = frames_list[:4] # Only use first 200 frames for overfitting

        return frames_list

    def _build_split_frames(self, splits):
        """
        Builds all split frames from scratch and saves them as .txt files with seq frame pairs
        Input:
            split (str): Split to build
        """

        frames_list = []
        annos_dir = self.depth_label_dir
        sequences = [seq for seq in os.listdir(annos_dir) 
                    if (seq not in self.skip_sequences and os.path.isdir(join(annos_dir, seq)))]
        sequences = sorted(sequences, key=lambda x: int(x))

        #2 Add all frames in a sequence to split
        for seq in sequences:
            cam_dir = join(annos_dir, seq, DEPTH_SUBDIRS[0]) # Assume stereo pairs both have depth

            frames = sorted([frame for frame in os.listdir(cam_dir) 
                        if join(cam_dir, frame).endswith(".png")], key=lambda x: int(os.path.splitext(x)[0]))

            for frame in frames:
                frames_list.append(f'{seq} {frame.split(".")[0]}')

        #2 Divide frames_list into splits randomly
        train_percent, val_percent, test_percent = 0.7, 0.15, 0.15
        num_frames   = len(frames_list)
        num_train       = int(num_frames * train_percent)
        num_val         = int(num_frames * val_percent)
        num_test        = int(num_frames * test_percent)

        indices = np.arange(0, len(frames_list), 1)
        rng.shuffle(indices)

        train, val, test    = indices[:num_train], indices[num_train:num_train+num_val], \
            indices[num_train+num_val:num_train+num_val+num_test]

        #3 Save splits to file
        frames_list = np.array(frames_list)
        for split in splits:
            split_path = join(annos_dir, f'{split}.txt')
            if "train" in split:
                np.savetxt(split_path, frames_list[train], fmt='%s')
            elif "val" in split:
                np.savetxt(split_path, frames_list[val], fmt='%s')
            elif "test" in split:
                np.savetxt(split_path, frames_list[test], fmt='%s')
            else:
                raise ValueError(f"Split {split} not recognized")

    def _load_data_paths(self, frames_list):
        """
        Loads the paths of the images, point cloud binaries, and calibration files.
        """
        self.calib_list = [] # Number of frames
        self.image_list = [] # Number of frames
        self.depth_list = [] # Number of frames
        self.pc_list    = [] # Number of frames
        self.pose_list  = [] # Number of frames
        self.label_list = [] # Number of frames

        calib_dir   = join(self.root_dir, CALIBRATION_DIR)
        image_dir   = join(self.root_dir, f'downsampled_{self.ds_rgb}')
        depth_dir   = join(self.root_dir, f'downsampled_{self.ds_rgb}', 
            f'{DEPTH_DIR}_1_LA_all'
        )
        label_dir   = self.depth_label_dir

        import time
        print("Running frames_list")
        start_time = time.time()

        def build_cam_tup(seq, frame):
            # Returns (cam0_path, cam1_path, cam0_depth_path, cam1_depth_path)
            rgbd_list = []
            for i in range(2):
                rgbd_list.append((
                    fn2path(image_dir, 
                        frame2fn(CAMERA_DIR, CAMERA_SUBDIRS[i], seq, frame, "png")
                    ),
                    join(
                        depth_dir, f'{seq}', DEPTH_SUBDIRS[i], f'{frame}.png'
                    )
                ))
            return rgbd_list
        
        def build_calib_tup(seq):
            # Returns (cam0_calib, cam1_calib)
            calib_list = []
            for i in range(2):
                cam = CAMERA_SUBDIRS[i]
                calib_list.append((
                    join(calib_dir, seq, f'calib_{cam}_intrinsics.yaml'),
                    join(calib_dir, seq, f'calib_os1_to_{cam}.yaml')
                ))
            return calib_list

        def build_label_tup(seq, frame):
            # Returns (cam0_label, cam1_label)
            label_list = []
            for i in range(2):
                label_list.append(
                    join(label_dir, seq, DEPTH_SUBDIRS[i], f'{frame}.png')
                )
            return label_list

        #1 Build rgbd paths
        self.image_list = [ build_cam_tup(finfo[0], finfo[1]) for finfo in frames_list ]

        #2 Build calib paths
        self.calib_list = [build_calib_tup(finfo[0]) for finfo in frames_list]

        #3 Build label paths
        self.label_list = [build_label_tup(finfo[0], finfo[1]) for finfo in frames_list]

        # for frame_tuple in frames_list:
        #     seq, frame = frame_tuple

        #     calib_pairs, image_pairs, label_pairs = [], [], []
        #     for cam in CAMERA_SUBDIRS:
        #         camintr_path    = join(calib_dir, seq, f'calib_{cam}_intrinsics.yaml')  
        #         cam2lidar_path  = join(calib_dir, seq, f'calib_os1_to_{cam}.yaml')
        #         assert os.path.exists(camintr_path) and os.path.exists(cam2lidar_path), \
        #             f'Calibration files not found for sequence {seq}'
        #         calib_pairs.append((camintr_path, cam2lidar_path))

        #         image_path = join(image_dir, cam, seq, f'{CAMERA_DIR}_{cam}_{seq}_{frame}.png')
        #         depth_path = join(depth_dir, seq, cam, f'{frame}.png')
        #         assert os.path.exists(image_path), f'Image not found for {image_path}'
        #         assert os.path.exists(depth_path), f'Depth not found for {depth_path}'
        #         image_pairs.append([image_path, depth_path])

        #         label_path = join(label_dir, seq, cam, f'{frame}.png')
        #         assert os.path.exists(label_path), f'Label not found for {label_path}'
        #         label_pairs.append(label_path)

        #     # Calibrations, images, labels
        #     self.calib_list.append(calib_pairs)
        #     self.image_list.append(image_pairs)
        #     self.label_list.append(label_pairs)
        print("Finished frames_list ", time.time() - start_time)

    def collate_fn(self, batch):
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
        # Stack images into a single tensor
        seq = [item['seq'] for item in batch]

        # Stack images into a single tensorbatch
        frame = [item['frame'] for item in batch]

        # Stack images into a single tensor
        # import pdb; pdb.set_trace()
        images = torch.cat([item['image'] for item in batch], axis=0)
        # item['image'] for item in batch]
        # images = torch.from_numpy(images)
        
        # Labels are stacked into a single tensor
        # B, _, H, Wbatch[0]['depth_label'].shape
        labels = torch.cat([item['depth_label'] for item in batch], axis=0)
        # labels = labels.view(-1, 2, self.GT_DEPTH_H, self.GT_DEPTH_W)
        

        # [item['label'] for item in batch]
        
        # Combine the batch elements into a single dictionary
        return {
            'seq': seq,
            'frame': frame,
            'image': images,
            'depth_label': labels
        }

    def __len__(self):
        """
        Assumes images and point clouds are the same length.
        """
        return len(self.image_list)

    def _load_image(self, idx):
        image_tensor_pair = torch.empty((0, 4, self.IMG_H, self.IMG_W), dtype=torch.float32)
        for image_pair in self.image_list[idx]:
            #2 Load image and depth
            try:
                rgb = cv2.imread(image_pair[0], -1).astype(np.uint8)
                depth = cv2.imread(image_pair[1], -1).astype(np.float32) # Pillow cant handle 16bit depth
            except:
                print("Error loading image", image_pair[0], image_pair[1])

            #3 Downsample image and depth
            rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            if self.do_augmentation:
                rgb = self.augmentation(rgb)
                # cv2.imwrite("testrgb.png", rgb_aug*255)
                
            depth = torch.from_numpy(depth).float().unsqueeze(0)
            rgbd = torch.cat((rgb, depth), axis=0).unsqueeze(0)
            image_tensor_pair = torch.cat((image_tensor_pair, rgbd), axis=0)

        return image_tensor_pair

    def _load_pose(self, idx):
        pose_np = self.pose_list[idx]
        homo_mat = np.eye(4)
        homo_mat[:3, :3] = R.from_quat([pose_np[5], pose_np[6], pose_np[7], pose_np[4]]).as_matrix()
        homo_mat[:3, 3] = pose_np[1:4]

        pose_dict = {
            'ts': pose_np[0],
            'lidar2global': homo_mat
        }

        return pose_dict
    
    def _load_label(self, idx):
        label_tensor = torch.empty((0, self.GT_DEPTH_H, self.GT_DEPTH_W), dtype=torch.float32)
        if self.annos_type=="Depth":
            for label_path in self.label_list[idx]:
                label_img = cv2.imread(label_path, -1).astype(np.float32) # Leave in mm
                depth = torch.from_numpy(label_img).float().unsqueeze(0)
                label_tensor = torch.cat((label_tensor, depth), axis=0)
        
        return label_tensor

    def _set_augmentation(self):
        self.augmentation = ColorJitterAndRandomCrop(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5
        )

    def __getitem__(self, idx):
        #0 Get sequence and frame from path
        seq, frame  = self.frames_list[idx]

        #2 Load rgbd image inputs, depth labels
        self._set_augmentation()
        rgbd = self._load_image(idx)
        label = self._load_label(idx)

        sample = {
            'seq': seq,
            'frame': frame,
            'image': rgbd, 
            'depth_label': label.unsqueeze(1) # 2 x H x W -> 2 x 1 x H x W [for view dim]
        }

        return sample

if __name__ == '__main__':
    print("------   CODataset Test   -----")
    cfg_path = './configs/dataset/coda_ds4.yaml'
    assert os.path.exists(cfg_path), f'Config file {cfg_path} does not exist'
    with open(cfg_path, 'r') as f:
        cfg_file = yaml.safe_load(f)
    cfg_file['infill_strat'] = 'LA'

    #0 Initialize tracker instance and GroundedSAM
    depth_dataset_all = CODatasetDepth(
        cfg=cfg_file,
        split="all",
        annos_type="Depth"
    )
    print("------   CODataset Initialized     ------")
    #2 Test basic loop through whole dataset
    
    dataloader = DataLoader(
        depth_dataset_all, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=depth_dataset_all.collate_fn
    )
    for i, batch in enumerate(dataloader):
        print(i, batch['image'].shape, batch['depth_label'].shape)

        # import cv2
        # cv2.imwrite(
        #     f'testrgb0.png', batch['image'][0, :3, :, :].permute(1,2,0).numpy()*255
        # )
        # cv2.imwrite(
        #     f'testrgb1.png', batch['image'][1, :3, :, :].permute(1,2,0).numpy()*255
        # )
        # Break after the first batch to keep the output short
        # if i == 0:
        #     break
