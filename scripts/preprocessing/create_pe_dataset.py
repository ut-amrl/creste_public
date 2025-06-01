"""
This dataset creates the necessary files for the PEFree dataset. It encodes all the data into a single pickle
file, which can be loaded using the `pickle` module in Python.

Pickle File Contents:
    - rgbd: RGBD image of the scene (H, W, 4)
    - overlap: indexes to other images in the dataset that overlap with the current image
    - gt_feats: Ground truth features for the scene (H*W, F)
    - cam2globalbev: Camera -> LiDAR -> Global SE(3) -> Global BEV patches (u,v) -> (x, y) indices
    - intrinsics: Camera intrinsics (K)
    - extrinsics: Camera extrinsics to LiDAR (T_cam_to_lidar)
    - frame_infos:
        - seq: Sequence number
        - frame: Frame number
"""

import os
from os.path import join
import pickle
import argparse
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import random

import cv2
import numpy as np

import torch
import torch.nn.functional as F

from scripts.preprocessing.build_dense_depth import get_frames_dict
import creste.datasets.coda_helpers as ch 
from creste.datasets.coda_utils import *
import creste.utils.geometry as geom

from creste.utils.feature_extractor import *

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='data/creste', help="Input directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default='data/creste', help="Output directory to save the dataset")
    parser.add_argument("--dataset_type", type=str, default='pefree', help="Type of dataset to create, [pefree, depth]")
    parser.add_argument("--frame_ds", type=int, default=5, help='Subsampling rate for frames, defaults to 2hz')
    parser.add_argument("--model_type", type=str, default='dino_vitb8', help='Model type to use for feature extraction')
    parser.add_argument("--feat_dim", type=int, default=64, help='Feature dimension to reduce to')
    parser.add_argument("--img_shape", type=str, default='1024,1224', help='Input image shape e.g."H,W"')
    parser.add_argument("--overlap", default=False, action='store_true', help='Compute overlapping frames')
    parser.add_argument("--cu", type=int, default=8, help='Crop down amount in pixels (u)')
    parser.add_argument("--cv", type=int, default=10, help='Crop down amount in pixels (u)')
    parser.add_argument('--seq_list', nargs='+', type=int, default=None, help="Sequence to convert, default is all")
    return parser.parse_args()

""" BEGIN CUDA MP Handles """
# def initialize_cuda_device():
#     # This will initialize CUDA context in each new process
#     if torch.cuda.is_available():
#         device_index = torch.cuda.current_device()  # Get default/current device index
#         torch.cuda.set_device(device_index)  # Set device by index
#     else:
#         print("CUDA is not available. Check your installation and GPU availability.")
def initialize_cuda_device():
    if torch.cuda.is_available():
        # Retrieve the number of GPUs available
        num_gpus = torch.cuda.device_count()
        
        # Create a list to store the memory available on each GPU
        memory_available = []
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            # Get the current memory usage stats for the GPU
            mem_stats = torch.cuda.memory_stats(i)
            # 'allocated_bytes.all.current' gives the currently allocated memory
            # 'reserved_bytes.all.current' gives the current reserved memory
            # 'active_bytes.all.current' gives the current active memory
            free_memory = torch.cuda.get_device_properties(i).total_memory - mem_stats['allocated_bytes.all.current']
            memory_available.append(free_memory)
        
        # Find the indices of the GPUs with the maximum free memory
        max_memory = max(memory_available)
        candidates = [i for i, mem in enumerate(memory_available) if mem == max_memory]
        
        # Randomly select among the GPUs with the most memory if there's a tie
        selected_gpu = random.choice(candidates)
        
        # Set the selected GPU as the current device
        torch.cuda.set_device(selected_gpu)
        # print(f"Selected GPU index {selected_gpu} with {max_memory} bytes of free memory")
    else:
        print("CUDA is not available. Check your installation and GPU availability.")
""" END CUDA MP Handles """


def get_overlap_single(inputs):
    pose_idx, db_poses = inputs

    return geom.get_overlapping_views(pose_idx, db_poses)

def load_dino_feat_single(inputs):
    feats_path, max_feats = inputs
    initialize_cuda_device()
    temp_feats = np.load(feats_path, mmap_mode="r", allow_pickle=True)

    if temp_feats.shape[0] > max_feats:
        idx = np.random.choice(temp_feats.shape[0], max_feats, replace=False)
        temp_feats = temp_feats[idx]

    temp_feats = torch.from_numpy(temp_feats).float().cuda()
    return temp_feats

def reduce_dino_rgb_single(inputs):
    initialize_cuda_device()

    in_path, out_path, rgb_path, output_shape = inputs
    descriptors = torch.from_numpy(np.load(in_path)).float()
    C, H, W = descriptors.shape
    descriptors = descriptors.permute(1, 2, 0).reshape(-1, C)
    
    # Normalize features per image for visualization
    feat_min = descriptors.min(dim=0)[0]
    feat_max = descriptors.max(dim=0)[0]
    descriptors = (descriptors - feat_min) / (feat_max - feat_min)

    # Reduce to 3 dimensional rgb
    reduction_mat, feat_color_min, feat_color_max = get_robust_pca(
        descriptors
    )
    lowrank_descriptor = descriptors @ reduction_mat
    lowrank_descriptor = lowrank_descriptor.reshape(
        H, W, -1 # because we are using indexing row and column
    ).squeeze()

    norm_lowrank_descriptor = ((lowrank_descriptor - feat_color_min) / (feat_color_max - feat_color_min)).clamp(0, 1)
    rgb_descriptor = (norm_lowrank_descriptor.cpu().numpy() * 255).astype(np.uint8)
    
    descriptor_img = Image.fromarray(rgb_descriptor)
    descriptor_img = descriptor_img.resize((output_shape[1], output_shape[0]), Image.NEAREST)
    descriptor_img_np = np.array(descriptor_img)

    img = Image.open(rgb_path).convert("RGB")
    ds_img = img.resize((output_shape[1], output_shape[0]), Image.NEAREST)
    img_np = np.array(ds_img)
    
    alpha = 0.2
    overlay = cv2.addWeighted(img_np, alpha, descriptor_img_np, 1 - alpha, 0)
    overlay = Image.fromarray(overlay)

    overlay.save(out_path)

def get_feat_minmax_single(inputs):
    feats_path, reduce_to_target_dim_mat = inputs
    feats = torch.from_numpy(np.load(feats_path)).float()
    reduced_feats = feats @ reduce_to_target_dim_mat
    return (reduced_feats.min(dim=0)[0], reduced_feats.max(dim=0)[0])

def postprocess_dino_single(inputs):
    initialize_cuda_device()
    try:
        in_path, out_path, reduce_to_target_dim_mat, feat_min, feat_max, input_shape, output_shape = inputs

        # Reduce features using PCA
        feats = torch.from_numpy(np.load(in_path, allow_pickle=True)).float()
        reduced_feats = feats.cuda() @ reduce_to_target_dim_mat.cuda()

        if feat_min is not None and feat_max is not None:
            reduced_feats = (reduced_feats - feat_min) / (feat_max - feat_min)
        reduced_feats = reduced_feats.reshape(
            input_shape[0], input_shape[1], -1
        ).permute(2, 0, 1) # [F, H, W]

        # Interpolate to output shape (bilinear)
        reduced_feats = F.interpolate(reduced_feats.unsqueeze(0), size=output_shape, mode='bilinear')
        reduced_feats = reduced_feats.squeeze().cpu().numpy()
        np.save(out_path, reduced_feats)
    # existing processing code
    except Exception as e:
        print(f"Error processing {inputs}: {e}")
        return None  # or another error indication

def resize_and_save_single(inputs):
    save_path, feats, output_shape = inputs
    feats = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0)
    feats = F.interpolate(feats, size=output_shape, mode='bilinear')
    feats = feats.squeeze().float().numpy()
    np.save(save_path, feats)

def reduce_and_save_single(inputs):
    in_path, out_path = inputs
    feats = torch.from_numpy(np.load(in_path)).float()

def save_pefree_pickle_single(inputs):
    """
    Save the PEFree pickle file for the given inputs.

    Output File: {outdir}/{seq}_{frame}.pkl
        seq: Sequence number
        frame: Frame number
        overlap: List of overlapping frame global indexes
    """
    indir, outdir, global_idx, seq, frame, gb_pose, overlap_dict = inputs

    calib_dict = {}

    #1 Load the camera intrinsics
    calib_dir = join(indir, CALIBRATION_DIR)
    calib_dict.update(ch.load_intrinsics(indir, seq, CAMERA_SUBDIRS[0]))

    #2 Load the camera to LiDAR extrinsics
    calib_dict.update(ch.load_extrinsics(indir, seq, CAMERA_SUBDIRS[0]))

    #3 Load the LiDAR to World extrinsics
    calib_dict['T_lidar_to_world'] = gb_pose

    #4 Load the RGBD path
    sample = {
        "id": f"{seq}_{frame}",
        "global_idx": global_idx,
        "calib": calib_dict,
        "pose": gb_pose
    }
    if overlap_dict is not None:
        sample.update(overlap_dict)

    pkl_path = join(outdir, f'{frame}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(sample, f)

def create_pefree_dataset(args):
    """
    Create the PEFree dataset metadata from the following directories. This function will create a global idx for each frame and overlap indexes for each frame. Then, it will call mp to create the individual pickle files that don't require global context.


    pose: Pose files, file structure poses/dense_global/{seq}.txt
    calibrations: calibration files calibrations/calib_cam0_intrinsics.yaml, calibrations/calib_os1_to_cam0.yaml
    """
    indir, outdir, model, pca_target_feature_dim = args.input_dir, args.output_dir, args.model_type, args.feat_dim
    frame_ds = args.frame_ds
    img_shape = ' '.join(args.img_shape.split(','))
    compute_overlap = args.overlap
    selected_seq_list = args.seq_list

    divisor = 1
    INPUT_SHAPES = {
        "dinov1": {
            "1024 1224": (1024-divisor*8, 1224-divisor*8),   
            "512 612": (512-divisor*8, 612-divisor*8)     
        },
        "dinov2": {
            "1024 1224": (1022 - divisor*14, 1218 - divisor*14),
            "512 612": (504 - divisor*14, 602 - divisor*14)
        }
    }
    DINO_OUTPUT_SHAPES = {
        "dinov1": {
            "1024 1224": (128, 153),
            "512 612": (63, 75)
        },
        "dinov2": {
            "1024 1224": (129, 153),
            "512 612": (69, 83)
        }
    }

    camid = 'cam0'
    sensor_dict = [
        ['2d_rect', camid]
    ]
    INPUT_SHAPE = INPUT_SHAPES[model][img_shape]
    DINO_OUTPUT_SHAPE = DINO_OUTPUT_SHAPES[model][img_shape]

    # model = "dinov1"
    if model == "dinov1":
        model_type = "dino_vitb8"
        stride=8
        # INPUT_SHAPE = (944, 1144)
        # DINO_OUTPUT_SHAPE = (118, 143) # (143, 118) # H W
        IMG_OUTPUT_SHAPE = (128, 153)
        NUM_WORKERS=24
    elif model == "dinov2":
        model_type =  "dinov2_vitb14"
        stride=7 # 8 for v1, and 7 for v2
        
        # INPUT_SHAPE = (910, 1078)
        # DINO_OUTPUT_SHAPE = (129, 153) # (153, 129) # H W
        IMG_OUTPUT_SHAPE = (128, 153) # (153, 128) # H W
        NUM_WORKERS=48
    else:
        raise ValueError("Invalid model type")

    extractor = ViTExtractor(
        model_type=model_type,
        stride=stride,
    )
    # Move extractor to cuda
    extractor = extractor.cuda()

    # run_steps = ["LOAD_PATHS", "COMPUTE_OVERLAP"]
    # run_steps = ["LOAD_PATHS", "GEN_FEATS", "REDUCE_FEATS", "VIS_FEATS"]
    run_steps = ["LOAD_PATHS", "COMPUTE_OVERLAP", "GEN_FEATS", "REDUCE_FEATS", "VIS_FEATS"]
    # run_steps = ["LOAD_PATHS", "COMPUTE_OVERLAP", "GEN_FEATS", "REDUCE_FEATS", "VIS_FEATS"]

    full_indir_list     = []
    full_infos_outdir_list    = []
    full_global_idx_list= []
    full_seq_list       = []
    full_frame_list     = []
    full_gb_pose_list   = []
    full_frames_path_list=[]
    full_dino_raw_list  = []
    full_dino_reduced_list  = []
    full_dino_rgb_list      = []

    # Load seq list from the available sequences
    seq_list = ch.get_available_sequences(indir)

    # Filter the sequence list by user selected
    if selected_seq_list is not None and len(selected_seq_list) > 0:
        seq_list = [seq for seq in seq_list if int(seq) in selected_seq_list]

    # seq_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21]
    # seq_list = [5, 6, 7]
    # seq_list = [9, 10, 11, 12]
    # seq_list = [0]
    print("Processing sequences: ", seq_list)
    if "LOAD_PATHS" in run_steps:
        dino_root_dir = join(outdir, f'{model_type}_raw')
        dino_reduced_root_dir = join(outdir, f'{model_type}_{pca_target_feature_dim}_reduced')
        for seq in seq_list:
            #1 Create the global idx for each frame
            frames_dict = get_frames_dict(args.input_dir, seq, sensor_dict)

            if "COMPUTE_OVERLAP" in run_steps:
                #2 Load the poses
                pose_path = join(indir, 'poses', 'dense_global', f'{seq}.txt')
                if not os.path.exists(pose_path):
                    pose_path = join(indir, 'poses', 'dense', f'{seq}.txt')
                gb_poses    = np.loadtxt(pose_path, dtype=np.float64) #Nx
                gb_poses    = ch.convert_poses_to_tf(gb_poses)
            else:
                ts_path = join(indir, 'timestamps', f'{seq}.txt')
                gb_poses = np.loadtxt(ts_path, dtype=np.float64) #Nx

            frames = [int(ch.get_info_from_filename(path, ext='jpg').split(' ')[-1])
                for path in frames_dict[camid]]
            subsampled_gb_poses = gb_poses[frames]
            subsampled_frames_path = frames_dict[camid]
            # subsampled_frames_path = frames_dict[camid][frames]

            #3 Prep auxiliary directories
            infos_outdir = join(outdir, 'infos', camid, f'{seq}')
            os.makedirs(infos_outdir, exist_ok=True)
            dino_seq_outdir = join(dino_root_dir, camid, f'{seq}')
            os.makedirs(dino_seq_outdir, exist_ok=True)
            dino_reduced_seq_outdir = join(dino_reduced_root_dir, camid, f'{seq}')
            os.makedirs(dino_reduced_seq_outdir, exist_ok=True)
            dino_rgb_seq_outdir = join(outdir, f'{model_type}_{pca_target_feature_dim}_rgb', camid, f'{seq}')
            os.makedirs(dino_rgb_seq_outdir, exist_ok=True)

            #4 Prep data for multiprocessing
            num_sub_frames  = len(subsampled_gb_poses)

            full_indir_list.extend([indir]*num_sub_frames)
            full_infos_outdir_list.extend([infos_outdir]*num_sub_frames)
            full_seq_list.extend([seq]*num_sub_frames)
            full_frame_list.extend(frames)
            full_frames_path_list.extend(
                subsampled_frames_path.tolist()
            )
            full_gb_pose_list.extend(subsampled_gb_poses)
            full_dino_raw_list.extend([dino_seq_outdir]*num_sub_frames)
            full_dino_reduced_list.extend([dino_reduced_seq_outdir]*num_sub_frames)
            full_dino_rgb_list.extend([dino_rgb_seq_outdir] * num_sub_frames)
        full_global_idx_list = np.arange(len(full_seq_list)).tolist()

    assert len(full_global_idx_list)==len(full_dino_raw_list)==len(full_gb_pose_list)==len(full_seq_list)==len(full_frame_list)==len(full_infos_outdir_list)==len(full_frames_path_list), "Lengths do not match"
    NUM_SAMPLES = len(full_global_idx_list) #TODO: Set this back after testing!

    if "COMPUTE_OVERLAP" in run_steps:
        print("Computing overlap for all frames")
        #3 Compute the overlapping pose set for each frame
        pool = mp.Pool(processes = NUM_WORKERS)
        full_gb_poses = np.array(full_gb_pose_list, dtype=np.float64)
        num_frames = len(full_gb_poses)
        overlap_list = []
        inputs = [(i, full_gb_poses) for i in range(num_frames)]
        if compute_overlap:
            with Pool(processes=NUM_WORKERS) as pool:
                with tqdm(total=len(inputs)) as pbar:
                    for result in pool.imap(get_overlap_single, inputs):
                        overlap_list.append(result)
                        pbar.update()
        else:
            overlap_list = [None] * num_frames

        """ BEGIN TESTING """
        # for input in inputs:
        #     overlap_list.append(get_overlap_single(input))
        #     break

        # save_pefree_pickle_single(
        #     (full_indir_list[0], full_infos_outdir_list[0], full_global_idx_list[0], full_seq_list[0], full_frame_list[0], full_gb_pose_list[0], overlap_list[0])
        # )
        """ END TESTING """

        #4 Pack data into lists for multiprocessing
        # overlap_list = [{} for _ in range(len(full_global_idx_list))]
        inputs = list(zip(full_indir_list, full_infos_outdir_list, full_global_idx_list, full_seq_list, full_frame_list, full_gb_pose_list, overlap_list))
        print("Saving pickle files with overlaps")

        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(inputs)) as pbar:
                for _ in pool.imap_unordered(save_pefree_pickle_single, inputs):
                    pbar.update()
    
    #5 Process the save raw Dino Features
    if "GEN_FEATS" in run_steps:
        print("Processing dino features: ")
        start_frame = 0
        end_frame = len(full_global_idx_list)
        full_global_idx_list = full_global_idx_list[int(start_frame):int(end_frame)]

        pbar = tqdm(full_global_idx_list, desc='Starting...')
        exp_dino_output_shape = DINO_OUTPUT_SHAPE[0] * DINO_OUTPUT_SHAPE[1]
        for global_idx in pbar:
            seq = full_seq_list[global_idx]
            frame = full_frame_list[global_idx]
            pbar.set_description(f'Processing seq: {seq}, frame: {frame}')
            img_path = full_frames_path_list[global_idx]
            dino_feats = extract_vit_features(extractor, img_path, INPUT_SHAPE) # FxHxW
            dino_feats = dino_feats.squeeze(0) # 1,P,F -> P,F
            assert dino_feats.shape[0] == exp_dino_output_shape, f"Invalid shape: {dino_feats.shape}"
            dino_feats = dino_feats.cpu().numpy()

            feat_dir = full_dino_raw_list[global_idx]
            np.save(join(feat_dir, f'{frame}.npy'), dino_feats)

    if "REDUCE_FEATS" in run_steps:
        print("Reducing features, normalizing and saving them.")
        # Two stage randomness to fit into memory
        torch.manual_seed(1337)

        #6a Load random subset of images
        N_frames = len(full_global_idx_list)

        max_images_to_load = min(int(0.1*NUM_SAMPLES), 500)
        max_features_per_image = 100
        full_dino_feats_paths = np.array([
            join(full_dino_raw_list[global_idx], f'{frame}.npy')
            for global_idx, frame in zip(full_global_idx_list, full_frame_list) 
        ], dtype=str)
        selected_feats_paths = np.random.choice(full_dino_feats_paths, max_images_to_load, replace=False)
        selected_feats_paths = [(path, max_features_per_image) for path in selected_feats_paths]
 
        dino_feats = None
        with Pool(processes=NUM_WORKERS) as pool:
            for result in tqdm(
                pool.imap(load_dino_feat_single, selected_feats_paths), 
                total=len(selected_feats_paths)
            ):
                result = result.cuda(0) # Move to gpu 0 by default
                if dino_feats is None:
                    dino_feats = result
                else:
                    dino_feats = torch.cat((dino_feats, result), dim=0)

        #6b Reduce the features using Robust PCA
        max_feats_to_compute_pca = min(100000, max_images_to_load * max_features_per_image)
        print(f'Reducing features using Robust PCA, randomly selecting {max_feats_to_compute_pca} features')
        reduce_to_target_dim_mat = compute_pca_reduction(
            dino_feats, max_feats_to_compute_pca, pca_target_feature_dim
        )
        reduce_to_target_dim_mat = reduce_to_target_dim_mat.cpu()
        
        #6c Obtain feat min and max across channel dim for all features move to batchwise
        # feat_min = torch.full((PCA_TARGET_FEATURE_DIM,), float('inf'))
        # feat_max = torch.full((PCA_TARGET_FEATURE_DIM,), float('-inf'))
        # feat_minmax_args = [
        #     (feats_path, reduce_to_target_dim_mat)
        #     for feats_path in full_dino_feats_paths
        # ]

        # feat_minmax_args = feat_minmax_args[:NUM_SAMPLES]

        # with Pool(processes=NUM_WORKERS) as pool:
        #     with tqdm(total=len(feat_minmax_args)) as pbar:
        #         for result in pool.imap_unordered(get_feat_minmax_single, feat_minmax_args):
        #             feat_min = torch.minimum(feat_min, result[0])
        #             feat_max = torch.maximum(feat_max, result[1])
        #             pbar.update()

        print("Normalizing and saving features after PCA reduction")
        #6d Reduce, normalize and save the features
        reduced_dino_feats_paths = [
            join(full_dino_reduced_list[global_idx], f'{frame}.npy')
            for global_idx, frame in zip(full_global_idx_list, full_frame_list)
        ]
        postprocess_dino_args = [
            (in_path, out_path, reduce_to_target_dim_mat, None, None, DINO_OUTPUT_SHAPE, IMG_OUTPUT_SHAPE)
            for in_path, out_path in zip(full_dino_feats_paths, reduced_dino_feats_paths)
        ]
        postprocess_dino_args = postprocess_dino_args[:NUM_SAMPLES]

        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(postprocess_dino_args)) as pbar:
                for _ in pool.imap(postprocess_dino_single, postprocess_dino_args):
                    pbar.update()

    #6d Dimensionality reduce the features to RGB for visualization
    if "VIS_FEATS" in run_steps:
        print("Visualizing features")
        # Load and visualize the features
        reduce_dino_rgb_args = [
            (join(in_dir, f'{frame}.npy'), join(out_dir, f'{frame}.png'), rgb_path, IMG_OUTPUT_SHAPE)
            for frame, in_dir, out_dir, rgb_path in zip(full_frame_list, full_dino_reduced_list, full_dino_rgb_list, full_frames_path_list)
        ]
        reduce_dino_rgb_args = reduce_dino_rgb_args[:NUM_SAMPLES]

        with Pool(processes=NUM_WORKERS) as pool:
            reduce_dino_rgb_single(reduce_dino_rgb_args[0])
            with tqdm(total=len(reduce_dino_rgb_args)) as pbar:
                for _ in pool.imap(reduce_dino_rgb_single, reduce_dino_rgb_args):
                    pbar.update()

    print("Done processing PEFree dataset")

def main(args):
    """
    Main function handles logic for deciding which dataset to create.
    """
    if args.dataset_type == 'pefree':
        create_pefree_dataset(args)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")


if __name__ == "__main__":
    mp.set_start_method('spawn')  # Set the start method for multiprocessing
    args = parse_args()
    main(args)
