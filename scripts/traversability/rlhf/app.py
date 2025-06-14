import os
import torch
import pickle
import yaml
from flask import Flask, jsonify, render_template, request
import numpy as np
from io import BytesIO
import base64
from PIL import Image
from scripts.traversability.planner_utils.control import (
    sampleTrajectory, 
    sampleEpsilonTrajectory, 
    transformToBEV,
    transformToLocal, 
    hausdorffDistance
)

# Custom Imports
import torch
from torch.utils.data import DataLoader
from creste.datasets.codapefree_dataloader import CodaPEFreeDataset
from creste.utils.visualization import (
    visualize_bev_poses, 
    visualize_bev_label, 
    visualize_rgbd_bev
)
from creste.datasets.coda_utils import *

import argparse
parser = argparse.ArgumentParser(description='Trajectory Labelling Tool')
parser.add_argument('--cfg_file', type=str,
                    default='configs/dataset/traversability/creste_sam2elevtraverse_horizon.yaml',
                    help='Path to the YAML data config file')
parser.add_argument('--split_dir', type=str, 
    default='data/creste/splits/3d_sam_3d_sam_dynamic_elevation_traversability_counterfactuals_standard', 
    help='Path to the dataset split directory')
parser.add_argument('--pickle_file', type=str, default=None, help='Path to the pickle file with offline evaluation metrics (optional) defaults to filtering by curvature')
parser.add_argument('--split', type=str, default='training',
                    help='Dataset split to use [training, validation, testing]')
parser.add_argument('--output_dir', type=str,
                    default='./data/creste/counterfactuals', help='Output directory')
parser.add_argument('--min_deviation', type=float, default=1.0,
                    help='Minimum deviation from straight line path for trajectory sampling')
parser.add_argument('--sampling_method', type=str, default='interpolation',
                    help='Method to sample trajectories [random, interpolation]')
parser.add_argument('--num_traj', type=int, default=10,
                    help="Number of trajectories to sample")
parser.add_argument('--skip_factor', type=int, default=1,
                    help='Number of samples to skip per step')
parser.add_argument('--view_frames', type=int, default=10,
                    help="Number of frames to use for rgb visualization")
parser.add_argument('--epsilon', type=float, default=2.5,
                    help='Max distance allow for trajectories to be shown to user')
parser.add_argument('--port', type=int, default=4242, help='Port number')
args = parser.parse_args()
assert os.path.exists(args.cfg_file), "Data config file not found"

app = Flask(__name__)

# Store the current trajectory index (global variable to keep track of progress)
trajectory_index = -1
trajectory_sample = None
dataloader = None  # Global dataloader to be initialized later
ACTION_HORIZON = 50

def get_sample(index):
    sample = dataloader[index]
    return sample


def se2_to_xytheta(pose):
    """Bx3x3 -> Bx3"""
    x = pose[:, 0, 2]
    y = pose[:, 1, 2]
    yaw = torch.atan2(pose[:, 1, 0], pose[:, 0, 0])
    return torch.stack((x, y, yaw), dim=1)


def get_info(sample):
    """ Returns sequence, frame for data sample"""
    seq, frame = sample['sequence'][0].item(), sample['frame'][0].item()
    return seq, frame


@app.route('/load', methods=['GET'])
def generate_trajectory():
    global trajectory_index
    global trajectory_sample
    global ACTION_HORIZON

    next_index = int(request.args.get('index', '0'))
    regen = int(request.args.get('regen', '0'))

    is_not_initialized = (next_index == -1) and (trajectory_index == -1)

    if is_not_initialized:
        next_index = 0
    elif next_index >= 0:
        pass
    elif trajectory_index >= 0:
        next_index = trajectory_index + args.skip_factor

    trajectory_index = next_index
    loaded_sample = get_sample(trajectory_index)
    trajectory_sample = loaded_sample

    epsilon = args.epsilon / 0.1  # Convert to BEV space

    map_res = 0.1
    if args.sampling_method == "random":
        traj_cfg = {
            "num_traj": args.num_traj,
            "num_iter": ACTION_HORIZON,
            "cmin": -2,
            "cmax": 2,
            "vmin": 0.1,
            "vmax": 0.75,
            "w": 1.0,
            "dt": 0.3,
            "epsilon": epsilon
        }
    elif args.sampling_method == 'interpolation':
        traj_cfg = {
            "num_traj": args.num_traj,
            "num_iter": ACTION_HORIZON,
            "num_samples": 2,
            "epsilon": args.epsilon,
        }
    else:
        raise ValueError("Invalid sampling method")

    expert = loaded_sample[TASK_TO_LABEL[TRAVERSE_LABEL_DIR]]
    expert_bev = np.ones((expert.shape[0], 2))
    expert_bev[:, :2] = expert[:, :2, 2]
    rgbd = loaded_sample['image']
    p2p = loaded_sample['p2p_in']

    seq, frame = get_info(trajectory_sample)
    label_file = os.path.join(args.output_dir, str(seq), f"{frame}.pkl")
    if os.path.exists(label_file) and not regen:
        with open(label_file, 'rb') as f:
            minibatch = pickle.load(f)
            labels = minibatch['rank']
            trajectories = minibatch['trajectories']
            # trajectory_index = minibatch['sample_idx']
            # print("Trajectories loaded from file ", trajectory_index)
            print(f"Loaded trajectories from {label_file}")
    else:
        num_traj_in_epsilon = 0
        total_traj = traj_cfg['num_traj']
        total_trajectories = np.zeros((0, traj_cfg["num_iter"], 2))
        while total_traj - num_traj_in_epsilon > 0:
            if args.sampling_method == "random":
                trajectories = sampleTrajectory(**traj_cfg)
            elif args.sampling_method == "interpolation":
                expert_traj = np.zeros((traj_cfg["num_iter"], 3))
                expert_traj[:, :2] = expert[:, :2, 2]
                traj_cfg['expert_traj'] = transformToLocal(expert_traj[None,...], res=map_res)
                trajectories = sampleEpsilonTrajectory(**traj_cfg)
            else:
                raise ValueError("Invalid sampling method")
            trajectories = transformToBEV(trajectories, res=map_res)
            candidate_trajectories = np.concatenate(
                (expert_bev[None, :, :2], trajectories), axis=0)
            
            distances = hausdorffDistance(candidate_trajectories, expert_idx=0)[
                1:]  # Exclude the expert
            epsilon_mask = distances < epsilon
            trajectories = trajectories[epsilon_mask]
            total_trajectories = np.concatenate(
                [total_trajectories, trajectories], axis=0)[:total_traj]

            num_traj_in_epsilon = total_trajectories.shape[0]
        labels = np.zeros(total_trajectories.shape[0]+1)

        trajectories = np.concatenate(
            (expert_bev[None, :, :2], total_trajectories), axis=0)

    bev_image_th = visualize_rgbd_bev(rgbd, p2p, map_res=0.1, map_sz=(
        256, 256), map_origin=(128, 128), num_scans=args.view_frames, num_cams=1)
    bev_image = bev_image_th.permute(1, 2, 0).cpu().numpy()
    bev_image_str = image_to_str(bev_image)
    front_image = rgbd[0, [0, 1, 2], :, :].permute(1, 2, 0).numpy()

    # Using the same function to convert front-view im
    front_image_str = image_to_str(front_image)
    # Return the trajectories and images to the frontend
    return jsonify({
        'seq': seq,
        'frame': frame,
        'trajectories': trajectories.tolist(),  # Convert NumPy arrays to list
        'labels': labels.tolist(),  # Optimal/suboptimal labels
        'bev_image': bev_image_str,  # Convert NumPy array to list
        'front_image': front_image_str,  # Convert NumPy array to list
        'index': trajectory_index  # Send the current trajectory index
    })

# Route for saving trajectories and their labels


@app.route('/save', methods=['POST'])
def save_trajectory():
    data = request.json
    trajectories = np.array(data['trajectories'])
    rank = np.array(data['labels'])  # (0 optimal, 1 suboptimal)

    seq, frame = get_info(trajectory_sample)
    # Save to local directory
    save_dir = os.path.join(args.output_dir, str(seq))
    save_file = os.path.join(save_dir, f"{frame}.pkl")
    os.makedirs(save_dir, exist_ok=True)

    minibatch = {
        "trajectories": trajectories,
        "rank": rank,
        "seq": seq,
        "frame": frame,
        "sample_idx": trajectory_index
    }
    # Save as pickle file
    with open(save_file, 'wb') as f:
        pickle.dump(minibatch, f)

    return jsonify({'status': 'success', "seq": seq, "frame": frame})


def image_to_str(image_array):
    # Convert NumPy array to image using PIL
    im = Image.fromarray((image_array * 255).astype('uint8'))
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Route for main web interface


@app.route('/')
def index():
    return render_template('index.html')


def init_dataloader(cfg):
    global ACTION_HORIZON
    ACTION_HORIZON = cfg['action_horizon']
    # labels = [SSC_LABEL_DIR, SOC_LABEL_DIR, ELEVATION_LABEL_DIR, TRAVERSE_LABEL_DIR]
    labels = [SAM_LABEL_DIR, SAM_DYNAMIC_LABEL_DIR, ELEVATION_LABEL_DIR, TRAVERSE_LABEL_DIR]
    sload_keys = ['p2p', 'fov_mask']
    sload_keys += [TASK_TO_LABEL[label] for label in labels]

    dataset = CodaPEFreeDataset(
        cfg=cfg,
        split=args.split,
        views=args.view_frames,
        skip_sequences=[],
        # skip_sequences=[8, 14, 15, 22],
        # skip_sequences=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        # 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22],
        fload_keys=['sequence', 'frame', 'image', 'depth_label', 'pose'],
        sload_keys=sload_keys,
        task_cfgs=cfg['task_cfgs'],
        # task_cfgs=[
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
            #     'name': SAM_LABEL_DIR,
            #     "kwargs": {
            #         "remap_labels": True,
            #         "subdir": "postprocess_rlang/build_map_outputs/sam2/static",
            #         "num_classes": 1, # Note remap=True will change the actual number of classes
            #         "kernel_size": 3,
            #         "ext": "npy"
            #     }
            # },
            # {
            #     'name': SAM_DYNAMIC_LABEL_DIR,
            #     "kwargs": {
            #         "remap_labels": True,
            #         "subdir": "postprocess_rlang/build_map_outputs/sam2/dynamic",
            #         "num_classes": 3,  # Note remap=True will change the actual number of classes
            #         "kernel_size": 5,
            #         "ext": "npy"
            #     }
            # },
            # {
            #     'name': ELEVATION_LABEL_DIR,
            #     "kwargs": {
            #         "subdir": "postprocess_rlang/build_map_outputs/geometric/elevation/labels",
            #         "num_classes": 2,
            #         "ext": "bin"
            #     }
            # },
            # {
            #     'name': TRAVERSE_LABEL_DIR,
            #     'kwargs': {
            #         'num_views': 50,
            #         'subdir': "data/coda_rlang/poses/dense_global",
            #         'num_classes': 0,  # dummy variable
            #         'ext': "txt"
            #     }
            # },
        # ],
        do_augmentation=False,
    )
    return dataset

def prep_split_dir(infos_dict):
    frame_infos = np.array([frame_info for frame_info in zip(infos_dict['sequences'], infos_dict['frames'])], dtype=int)
    distances = np.array([info['hausdorff_distances'][0] for info in infos_dict['non_serializable']['infos']],
        dtype=np.float32)
    assert len(frame_infos) == len(distances), "Frame info and distances do not match"

    # Sort by distance
    sorted_indices = np.argsort(distances)
    frame_infos = frame_infos[sorted_indices]
    distances = distances[sorted_indices]

    # Get average hdist
    avg_hdist = np.mean(distances)

    # If greater 1 std above the mean, add to the split
    std_hdist = np.std(distances)
    high_deviation_frames = frame_infos[distances >= avg_hdist + std_hdist]

    # Sort by seq frame
    high_deviation_frames = high_deviation_frames[np.lexsort((high_deviation_frames[:, 1], high_deviation_frames[:, 0]))]

    # Only keep unique (seq, frame tuples)
    keep_frames = [f'{frame_info[0]} {frame_info[1]}' for frame_info in high_deviation_frames]
    keep_frames = np.unique(keep_frames)
    keep_frames_split = np.array([frame.split() for frame in keep_frames], dtype=int)

    split_dir = os.path.join(args.split_dir, 'counterfactuals')
    if not os.path.exists(split_dir):
        print("Creating split directory ", split_dir)
        os.makedirs(split_dir)
    split_file = os.path.join(split_dir, f'{args.split}.txt')
    np.savetxt(split_file, keep_frames_split, fmt='%d')

    print(f"Saved frames exceeding 1 std dev to: {split_file}")
    print(f"Overriding split_dir in cfg with new split dir: {split_dir}")
    args.split_dir = split_dir

# /robodata/arthurz/Research/lift-splat-map/model_outputs/TraversabilityLearning/depth128UD_dinov1pretrain_sam2dynelev_supcon_joint_BB_efficientnet-b0_Head_depthconv-head_lr_0.000500_UD_LAIDW_v2/creste_terrainnet_dinopretrain_maxentirl_msfcn_sam2dynsemelev_headMaxEntIRL_horizon50/20241215/013033/all/epoch25/traversability_outputs.pkl
if __name__ == '__main__':
    os.chdir(os.path.join(os.getcwd(), ""))
    if args.pickle_file is not None:
        with open(args.pickle_file, 'rb') as f:
            offline_metrics = pickle.load(f)
            prep_split_dir(offline_metrics)

    # Load trajectories on startup
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg['resample_trajectories'] = False
        cfg['datasets'][0]['split_dir'] = args.split_dir
    cfg['min_deviation'] = args.min_deviation
    dataloader = init_dataloader(cfg)

    # Start the Flask web server
    app.run(debug=True, port=args.port)