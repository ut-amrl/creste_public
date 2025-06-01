"""
This script selects samples that are present for all labelled tasks.
It postprocesses samples by filtering out those where the hausdorff distance
between the start and end poses is greater than a threshold. The end pose must
be in front of the start pose. The samples are then balanced between curved and
straight paths and saved as train/val/test splits.
"""
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from os.path import join
import glob
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.transform import Rotation as R

from creste.utils.utils import drop_overlapping_horizons
from creste.datasets.coda_utils import *
import creste.datasets.coda_helpers as ch
from creste.datasets.codapefree_dataloader import CodaPEFreeDataset

DEFAULT_TRAIN_TASKS = [SAM_LABEL_DIR, SAM_DYNAMIC_LABEL_DIR, ELEVATION_LABEL_DIR, TRAVERSE_LABEL_DIR]
DEFAULT_EVAL_TASKS = [SSC_LABEL_DIR, SOC_LABEL_DIR]

DEFAULT_TRAIN_STR = ' '.join(sorted(DEFAULT_TRAIN_TASKS))

np.random.seed(1337)

def parse_args():
    parser = argparse.ArgumentParser(description='Create splits for dataset')
    parser.add_argument('--cfg_file', 
        default="configs/dataset/traversability/coda_ds2_sam2elevtraverse_horizon50.yaml", 
        type=str, help='Path to config file')
    parser.add_argument('--hausdorff', default=1.0, type=float, help='Hausdorff distance threshold')
    parser.add_argument('--horizon', default=100, type=int, help='Number of frames to consider for filtering (@10Hz)')
    parser.add_argument('--overlap', default=10, type=int, help='Number of frames to overlap between horizons')
    parser.add_argument('--min_distance', default=3.0, type=float, help='Minimum distance moved')
    parser.add_argument('--out_dir', default="data/coda_pefree_rlang/splits", type=str, help='Output directory')
    parser.add_argument('--split_type', default='curvature', type=str, help='Type of split to create [standard, curvature]')
    args = parser.parse_args()

    return args

def frames_sort_func(x):
    return int(x.split(' ')[0])*100000 + int(x.split(' ')[1])

def load_traversability_samples(task_cfg, skip_sequences):
    subdir = task_cfg['subdir']
    ext = task_cfg['ext']

    action_horizon = task_cfg['num_views']
    seq_subpaths = [f for f in sorted(glob.glob(
        join(subdir, f'*.{ext}')), key=lambda x: ch.pose_sort_func(x))
        if os.path.basename(f).split('_')[0] not in skip_sequences]

    task_infos_np = np.empty((0, 1), dtype=str)
    for seq_subpath in seq_subpaths:
        f = np.loadtxt(seq_subpath)
        seq = os.path.basename(seq_subpath).split('.')[0]
        frames = np.arange(0, len(f), 5)
        infos = np.array([f"{seq} {frame}" for frame in frames], dtype=str).reshape(-1, 1)
        task_infos_np = np.concatenate([task_infos_np, infos], axis=0)
    task_infos_np = task_infos_np.flatten()
    task_infos_np = drop_overlapping_horizons(
        task_infos_np, args.overlap)

    return task_infos_np

def load_samples(cfg):
    """
    For each task, load the valid frames. Then intersect the valid frames for each task
    to get the final valid frames for the dataset.
    """
    task_infos = {}
    task_cfgs_list = cfg['task_cfgs']
    skip_sequences = cfg['skip_sequences']
    for task_cfg_dict in task_cfgs_list:
        task = task_cfg_dict['name']
        task_cfg = task_cfg_dict['kwargs']

        if task == 'counterfactuals':
            print("Skipping counterfactuals")
            continue
        elif task == 'distillation':
            task_cfg['subdir'] = 'data/creste/depth_0_LA_all'
            task_cfg['ext'] = 'png'

        subdir = task_cfg['subdir']
        ext = task_cfg['ext']
        if not os.path.exists(subdir):
            print(f"Skipping {task} as {subdir} does not exist")
            continue

        if task == 'traversability':
            task_infos_np = load_traversability_samples(task_cfg, skip_sequences)
        else:
            # Load valid frames for each task
            seq_subdirs = ch.get_sorted_subdirs(
                subdir, exclude_dirs=skip_sequences)

            assert len(seq_subdirs) > 0, f"No sequences found in {subdir}"

            task_infos_np = np.concatenate(
                [ch.get_dir_frame_info(seq_subdir, ext=ext, short=True) for seq_subdir in seq_subdirs], axis=0
            )

        task_infos[task] = set(task_infos_np)

    task_info_sets = {task: set(task_infos[task]) for task in task_infos}
    common_elements = set.intersection(*task_info_sets.values())

    # Sort the common elements
    common_elements = sorted(list(common_elements), 
        key=frames_sort_func)
    return common_elements

def filter_by_poses(cfg, samples, horizon=100, min_distance=3): # 7 seconds horizon
    """This function filters the common elements by those where the hausdorff distance
    between the start and end pose comopares to the true path is greater than a threshold"""
    poses_dir = join(cfg['root_dir'], POSES_DIR, POSES_SUBDIRS[0])
    poses_dir = join(cfg['root_dir'], POSES_DIR, POSES_SUBDIRS[1]) if not os.path.exists(poses_dir) else poses_dir
    pose_paths = sorted(glob.glob(join(poses_dir, '*.txt')), key=ch.pose_sort_func)
   
    poses_dict = {os.path.basename(p).split('.txt')[0]: np.loadtxt(p, dtype=np.float64) for p in pose_paths}

    valid_samples = []
    hausdorff_dists = []
    total_traversal_distance = 0.0
    total_frame_count = 0.0
    for sample in samples:
        seq, frame = sample.split(' ')
        num_frames = min(horizon, len(poses_dict[seq])-int(frame))
        pose_window = poses_dict[seq][int(frame):int(frame)+num_frames]
        if num_frames < 0 or num_frames < horizon:
            print(f"Skipping {sample} as not enough frames")
            continue
        T_lidar_to_world = np.tile(np.eye(4), (num_frames, 1, 1))
        quat = np.stack([pose_window[:, 5], pose_window[:, 6], pose_window[:, 7], pose_window[:, 4]], axis=1)
        
        T_lidar_to_world[:, :3, :3] = R.from_quat(quat).as_matrix()
        T_lidar_to_world[:, :3, 3] = pose_window[:, 1:4]

        T_init_to_world = T_lidar_to_world[0]
        T_world_to_init = np.linalg.inv(T_init_to_world)
        T_lidar_to_init = np.matmul(T_world_to_init, T_lidar_to_world)

        xy = T_lidar_to_init[:, :2, 3]
        # Ensure final position is in front of the initial position
        if xy[-1, 0] < xy[0, 0]:
            continue

        # Total displacement
        total_displacement = np.linalg.norm(xy[-1] - xy[0])
        if total_displacement < min_distance:
            continue
        
        # Ensure moved at least x meters from pose to pose
        diffs = np.diff(xy, axis=0)  # Compute differences between consvecutive poses
        total_distance = np.sum(np.linalg.norm(diffs, axis=1))  # Sum the norms of these differences
        total_traversal_distance += total_distance
        total_frame_count += num_frames

        # # Average velocity
        # avg_velocity = total_distance / (num_frames/2)

        # if avg_velocity < 0.3:
        #     continue

        # Compute the hausdorff distance between the straight line path and the actual path
        straight_line_path = np.linspace(xy[0], xy[-1], num_frames)
        forward_hd = directed_hausdorff(xy, straight_line_path)[0]
        backward_hd = directed_hausdorff(straight_line_path, xy)[0]
        hausdorff_distance = max(forward_hd, backward_hd)

        hausdorff_dists.append(hausdorff_distance)
        valid_samples.append(sample)

        # valid_samples.append({'sample': sample, 'hausdorff_dist': hausdorff_distance})
        # # Plot the pose xys and the straight line path
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(-xy[:, 1], xy[:, 0], 'r')
        # plt.plot(-straight_line_path[:, 1], straight_line_path[:, 0], 'b')
        # plt.xlim(-12.8, 12.8)
        # plt.ylim(-12.8, 12.8)
        # plt.savefig('test.png')
        # plt.close()
        # import pdb; pdb.set_trace()
   
    print(f"Average velocity over samples: {total_traversal_distance/(total_frame_count/2 + 1e-6)}")

    return np.array(valid_samples), np.array(hausdorff_dists)

def save_samples(csamples, cdistances, ssamples, sdistances, out_dir):
    # Divide into train/val/test splits
    num_curved_samples = len(csamples)
    num_straight_samples = len(ssamples)
    ctrain_split = int(0.7*num_curved_samples)
    cval_split = int(0.15*num_curved_samples)
    ctest_split = num_curved_samples - ctrain_split - cval_split
    strain_split = int(0.7*num_straight_samples)
    sval_split = int(0.15*num_straight_samples)
    stest_split = num_straight_samples - strain_split - sval_split

    if len(ssamples) == 0:
        samples_list = [
            csamples[:ctrain_split],
            csamples[ctrain_split:ctrain_split+cval_split],
            csamples[ctrain_split+cval_split:],
            csamples
        ]
        distances_list = [
            cdistances[:ctrain_split],
            cdistances[ctrain_split:ctrain_split+cval_split],
            cdistances[ctrain_split+cval_split:],
            cdistances
        ]
    else:
        samples_list = [
            csamples[:ctrain_split] + ssamples[:strain_split],
            csamples[ctrain_split:ctrain_split+cval_split] + ssamples[strain_split:strain_split+sval_split],
            csamples[ctrain_split+cval_split:] + ssamples[strain_split+sval_split:],
            csamples + ssamples
        ]
        distances_list = [
            cdistances[:ctrain_split] + sdistances[:strain_split],
            cdistances[ctrain_split:ctrain_split+cval_split] + sdistances[strain_split:strain_split+sval_split],
            cdistances[ctrain_split+cval_split:] + sdistances[strain_split+sval_split:],
            cdistances + sdistances
        ]

    for samples_split, distances_split, split_name in zip(samples_list, distances_list,['training', 'validation', 'testing', 'full']):
        # Sort by seq frame
        sample_ids = np.array([info.split(' ') for info in samples_split], dtype=int)
        sample_keys = np.lexsort((sample_ids[:, 1], sample_ids[:, 0]))
        samples_split = np.array(samples_split)[sample_keys]
        distances_split = np.array(distances_split)[sample_keys]

        split_save_path = join(out_dir, f'{split_name}.txt')
        np.savetxt(split_save_path, samples_split, fmt='%s')
        split_save_path = join(out_dir, f'{split_name}_distances.txt')
        np.savetxt(split_save_path, distances_split, fmt='%f')

    print(f"Saved splits to {out_dir}")

"""
python tools/build_splits.py --cfg_file configs/dataset/traversability/coda_ds2_sam2elevtraverse_horizon50.yaml
"""

def main(args):
    assert os.path.exists(args.cfg_file), f'Config file {args.cfg_file} does not exist'
    with open(args.cfg_file, 'r') as f:
        cfg_file = yaml.safe_load(f)

    threshold = args.hausdorff
    horizon = args.horizon
    split_type = args.split_type
    min_distance = args.min_distance
    
    # Load samples filtered by sequences and frames
    common_samples = load_samples(cfg_file)

    # Create sample directories
    split_dir_name = '_'.join([task_cfg['name'] for task_cfg in cfg_file['task_cfgs']])
    if split_type == 'curvature':
        split_dir_name += f'_hausdorff{int(threshold)}m_horizon{int(horizon)}_curvature'
    elif split_type == 'standard':
        split_dir_name += f'_standard'
    else:
        raise ValueError(f"Split type {split_type} not supported")
    out_dir= join(args.out_dir, split_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Compute hausdorff distances for the samples
    valid_samples, hausdorff_dists = filter_by_poses(cfg_file, common_samples, horizon=horizon, min_distance=min_distance)
    # Filter by hausdorff distances
    curved_samples = np.array(valid_samples)[hausdorff_dists > threshold]
    cs_distances = hausdorff_dists[hausdorff_dists > threshold]
    straight_samples = np.array(valid_samples)[hausdorff_dists <= threshold]
    ss_distances = hausdorff_dists[hausdorff_dists <= threshold]
    str
    print(f"Curved samples: {len(curved_samples)}")
    print(f"Straight samples: {len(straight_samples)}")

    # Balance number of curved and straight samples
    num_curved_samples = len(curved_samples)
    num_straight_samples = len(straight_samples)
    print(f"Sampled {num_curved_samples} curved samples and {num_straight_samples} straight samples")

    # Randomly select indices keeping things in order
    if split_type == 'standard':        
        all_samples = curved_samples.tolist() + straight_samples.tolist()
        # Shuffle the samples
        all_indices = np.arange(len(all_samples))
        np.random.shuffle(all_indices)
        all_samples = np.array(all_samples)[all_indices]
        all_distances = np.hstack([cs_distances, ss_distances])[all_indices]
        save_samples(all_samples, all_distances.tolist(), [], [], out_dir)
    else:
        curved_indices = np.arange(num_curved_samples)
        straight_indices = np.arange(num_straight_samples)
        np.random.shuffle(curved_indices)
        np.random.shuffle(straight_indices)
        curved_samples = curved_samples[curved_indices].tolist()
        curved_distances = cs_distances[curved_indices]
        straight_samples = straight_samples[straight_indices].tolist()
        straight_distances = ss_distances[straight_indices]
        all_samples = curved_samples + straight_samples
        all_distances = np.concatenate([curved_distances, straight_distances])

        save_samples(curved_samples, curved_distances.tolist(), straight_samples, straight_distances.tolist(), out_dir)

    # Plot sample distance distribution
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(all_distances, bins=20)
    plt.title('Hausdorff distances')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.savefig(join(out_dir, 'hausdorff_distances.png'))

if __name__ == "__main__":
    args = parse_args()
    main(args)