import os
from os.path import join
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import json
import multiprocessing
from multiprocessing import Pool

from scipy.spatial.transform import Rotation as R

import argparse

from creste.utils.depth_utils import compute_stereo_depth_map, compute_accum_lidar_depth_map, compute_filter_depth_map
from creste.utils.visualization import save_depth_image
from creste.utils.projection import pixels_to_depth
from creste.datasets.coda_utils import *
import creste.datasets.coda_helpers as ch

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy/15641148#15641148
# os.system("taskset -p 0xff %d" % os.getpid())

COMPUTE_DEPTH_INPUTS_LIST = []

# Single Frame
# python tools/build_dense_depth.py --cfg_file configs/dataset/creste.yaml --out_dir ./ --scans --proc LA --dataset_type single
def parse_args():
    parser = argparse.ArgumentParser(
        description='Build semantic map from point clouds')
    parser.add_argument('--cfg_file', type=str,
                        default='./configs/dataset/creste.yaml')
    parser.add_argument('--out_dir', type=str, default='./data/creste')
    parser.add_argument('--scans', type=int, default=50,
                        help="Number of scans to use")
    parser.add_argument('--verbose', default=False,
                        action='store_true', help="Saves debug images")
    parser.add_argument('--proc', type=str, default='IDW',
                        help="Postprocessing method [IDW, LA, LAIDW, PASS, STEREO]")
    parser.add_argument('--dataset_type', type=str, default='semistatic',
                        help="Postprocessing method [single, semistatic, semanticsegmentation, object, seq, all]")
    parser.add_argument('--ds_factor', type=int, default=5,
                        help="Downsampling factor for frames [Only works when dataset_type='seq']")
    parser.add_argument('--seq_list', nargs='+', type=int, default=None, help="Sequence to convert, default is all")
    args = parser.parse_args()

    return args


def get_frames_dict(root_dir, seq, sensor_dict, ds=1, start_frame=0):
    """
    Gets frames list for all sensors in a given sequence

    Inputs:
    root_dir: root directory of the dataset
    seq: sequence number (int)
    sensor_dict: list of sensor names and their corresponding directories
            [
                ['2d_rect', 'cam0'],
                ['2d_rect', 'cam1'],
                ['3d_raw', 'os1']
            ]

    Returns:
        frames_dict: dictionary of frames list for each sensor
            {
                'cam0': ['path/to/frames/000000.png', 'path/to/frames/000001.png', ...],
                'cam1': ['path/to/frames/000000.png', 'path/to/frames/000001.png', ...],
                'os1': ['path/to/frames/000000.png', 'path/to/frames/000001.png', ...]
            }
    """
    frames_dict = {}
    for sensor_info in sensor_dict:
        modality, name = sensor_info

        frames_dir = join(root_dir, modality, name, str(seq))
        assert os.path.exists(
            frames_dir), f'Frames directory {frames_dir} does not exist'

        frames_list = np.array(
            [int(os.path.splitext(f)[0].split('_')[-1]) for f in os.listdir(frames_dir)])
        sorted_frames_idx = np.argsort(frames_list)

        frames_path_list = np.array([join(frames_dir, f)
                                     for f in os.listdir(frames_dir)])
        frames_path_list = frames_path_list[sorted_frames_idx]

        # Apply frame downsampling
        frames_path_list = frames_path_list[::ds]
        frames_path_list = frames_path_list[start_frame:]

        frames_dict[name] = frames_path_list

    return frames_dict


def get_frames_from_json(root_dir, sensor_dict, override_all=False, ds=1, json_subpath="depth.json"):
    """
    Load all frames each sublist from a json file

    Inputs:
        root_dir: root directory of the dataset
        seq: sequence number (int)
        sensor_dict: list of sensor names and their corresponding directories
            [
                ['2d_rect', 'cam0'],
                ['2d_rect', 'cam1'],
                ['3d_raw', 'os1']
            ]
        json_subpath: subpath to json file from root_dir
    """
    if not override_all:
        json_path = join(root_dir, json_subpath)
        assert os.path.exists(
            json_path), f'JSON file {json_path} does not exist'

        seq_dict = json.load(open(json_path, 'r'))
    else:
        seq_dict = {}
        cam_dir = join(root_dir, CAMERA_DIR, CAMERA_SUBDIRS[0])
        seq_list = sorted([int(x) for x in os.listdir(cam_dir)])
        for seq in seq_list:
            frames_dir = join(cam_dir, str(seq))
            filenames = os.listdir(frames_dir)
            frames = [int(os.path.splitext(f)[0].split('_')[-1])
                      for f in os.listdir(frames_dir)]

            frames = sorted(frames, key=lambda x: int(x))
            print("Sequence", seq, "Frames", len(frames))
            # seq_dict[seq] = [[0, len(frames)]]
            seq_dict[seq] = [frames]

    frames_dict = {}
    for seq, frame_ranges in seq_dict.items():
        frames_dict[seq] = {}

        frames_list = []
        for frame_range in frame_ranges:
        #     # frames_list += list(range(frame_range[0], frame_range[1], ds))
            frames_list += frame_range[::ds]

        for sensor_info in sensor_dict:
            modality, name = sensor_info
            frames_dir = join(root_dir, modality, name, str(seq))
            if not os.path.exists(frames_dir):
                print(
                    f'Frames directory {frames_dir} does not exist, skipping annotations')
                continue

            sorted_frames_idx = np.sort(frames_list)

            fileext = os.listdir(frames_dir)[0].split('.')[-1]
            def fname_func(x): return f'{modality}_{name}_{seq}_{x}.{fileext}'
            frames_path_list = np.array([
                join(frames_dir, fname_func(f)) for f in sorted_frames_idx])

            frames_dict[seq][name] = frames_path_list

    return frames_dict


def get_frames_from_metadata(root_dir, sensor_dict, annotation_type="SemanticSegmentation"):
    """
    Load all frames from each metadata json file
    Inputs:
        root_dir: root directory of the dataset
        sensor_dict: list of sensor names and their corresponding directories
            [
                ['2d_rect', 'cam0'],
                ['2d_rect', 'cam1'],
                ['3d_raw', 'os1']
            ]
    Outputs:
        frames_dict: dictionary of frames list for each sensor
            {
                'cam0': ['path/to/frames/000000.png', 'path/to/frames/000001.png', ...],
                'cam1': ['path/to/frames/000000.png', 'path/to/frames/000001.png', ...],
                'os1': ['path/to/frames/000000.png', 'path/to/frames/000001.png', ...]
            }
    """
    metadata_dir = join(root_dir, METADATA_DIR)
    assert os.path.exists(
        metadata_dir), f'Metadata directory {metadata_dir} does not exist'

    seq_list = sorted([file.split('.')[0]
                       for file in os.listdir(metadata_dir)], key=lambda x: int(x))

    frames_dict = {}
    for seq in seq_list:
        frames_dict[seq] = {}

        meta_path = join(metadata_dir, f'{seq}.json')
        remapped_split = ANNOTATION_TYPE_MAP[annotation_type]
        obj_splits = json.load(open(meta_path, 'r'))[remapped_split]

        frames_list = []
        for split in obj_splits.keys():  # Per split
            frames = [os.path.basename(f).split('.')[0].split(
                '_')[-1] for f in obj_splits[split]]
            frames_list.extend(frames)
        frames_list = sorted(frames_list, key=lambda x: int(x))

        # Save frames list for each sensor
        for sensor_info in sensor_dict:
            modality, name = sensor_info
            frames_dir = join(root_dir, modality, name, seq)
            if not os.path.exists(frames_dir):
                print(
                    f'Frames directory {frames_dir} does not exist, skipping annotations')
                continue

            fileext = os.listdir(frames_dir)[0].split('.')[-1]
            def fname_func(x): return f'{modality}_{name}_{seq}_{x}.{fileext}'
            frames_path_list = np.array([
                join(frames_dir, fname_func(f)) for f in frames_list])

            frames_dict[seq][name] = frames_path_list
    return frames_dict


def load_pc_frames_time(frames_list, frame_idx, horizon):
    """
    Loads point cloud numpy arrays for a given frame index and horizon in frames_list

    Return a list of point clouds
    """
    pc_list = []
    # accum_frames_list = range(frame_idx - horizon//2, frame_idx + horizon//2 + 1)
    accum_frames_list = range(frame_idx - horizon, frame_idx+1)
    for i in accum_frames_list:
        # assert i < len(frames_list), f'Frame index {i} out of bounds for frames_list of length {len(frames_list)}'
        if i < 0 or i >= len(frames_list):
            # If frame index is out of bounds, return empty point cloud
            # pc = np.zeros((POINTS_PER_SCAN, 3), dtype=np.float32)
            # pc_list.append(pc)
            pc_list.append(None)
            continue

        # if i < 0:
        #     pc = np.zeros((POINTS_PER_SCAN, 3), dtype=np.float32)
        # else:
        pc_path = frames_list[i]
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(POINTS_PER_SCAN, -1)
        pc = pc[:, :3]
        pc_list.append(pc)

    return pc_list, accum_frames_list


def load_pc_frames_pose(frames_list, pose_np, frame_idx, horizon, dist_res=0.1):
    """
    Loads point cloud numpy arrays in increments of dist from each frame idx

    """
    pc_list = []
    accum_frames_list = []

    # 1 Greedily searches for point clouds in increments of dist from frame_idx
    ref_pose = pose_np[frame_idx]

    dist_list = np.arange(0, horizon*dist_res, dist_res)
    print(dist_list)
    for dist in dist_list:
        # Get closest pose previous in time
        search_pose_np = pose_np[:frame_idx]

        pose_dists = np.linalg.norm(
            search_pose_np[:, 1:4] - ref_pose[1:4], axis=1) - dist
        closest_pose_idx = np.argmin(np.abs(pose_dists))

        closest_pc_path = frames_list[closest_pose_idx]

        pc = np.fromfile(closest_pc_path, dtype=np.float32).reshape(-1, 4)
        accum_frames_list.append(closest_pose_idx)
        pc_list.append(pc)

    return pc_list, accum_frames_list


def pose_as_matrix(pose):
    """
    Converts pose to transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat([pose[5], pose[6], pose[7], pose[4]]).as_matrix()
    T[:3, 3] = pose[1:4]
    return T


def transform_pc_frames(pc_list, pose_np, ref_frame, accum_frames_list):
    """
    Transforms points clouds to the current ref_frame using pose_np
    pc_list - list of point clouds (Nx4)
    pose_np - numpy array of poses (Nx7)
    ref_frame- index of frame in the sequence to use as reference
    horizon - number of frames to use
    """
    T_ref_global = pose_as_matrix(pose_np[ref_frame])
    pc_ref_list = []

    for idx, frame_idx in enumerate(accum_frames_list):
        pc = pc_list[idx]
        if pc is None:
            continue

        # Get transformation matrix
        T_ego_global = pose_as_matrix(pose_np[frame_idx])

        # Manually construct inverse to avoid openblas call
        T_ref_global_inv = np.eye(4)
        T_ref_global_inv[:3, :3] = T_ref_global[:3, :3].T
        T_ref_global_inv[:3, 3] = -T_ref_global[:3, :3].T @ T_ref_global[:3, 3]
        T_ego_ref = T_ref_global_inv @ T_ego_global

        # Transform point cloud
        pc_ego = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))
        if pc_ego.shape[1] != 4:
            # raise ValueError(f'Point cloud shape {pc_ego.shape} is not valid, expected Nx3 or Nx4')
            continue

        try:
            pc_ref = (T_ego_ref @ pc_ego.T).T
        except Exception as e:
            print("Shape T_ego_ref :", T_ego_ref.shape)
            print("Shape pc_ego :", pc_ego.shape)
            print("Shape T_ego_global: ", T_ego_global.shape)
            print("frame_idx: ", frame_idx)
            raise e
        pc_ref_list.append(pc_ref[:, :3])
    return pc_ref_list


def load_calib(root_dir, seq, camid):
    calib_dir = join(root_dir, 'calibrations', str(seq))
    camintr_path = join(calib_dir, f'calib_{camid}_intrinsics.yaml')
    cam2lidar_path = join(calib_dir, f'calib_os1_to_{camid}.yaml')
    camintr_file = yaml.safe_load(open(camintr_path, 'r'))
    lidar2cam_file = yaml.safe_load(open(cam2lidar_path, 'r'))

    calib_dict = {
        'K': np.array(camintr_file['camera_matrix']['data']).reshape(3, 3),
        'd': np.array(camintr_file['distortion_coefficients']['data']),
        'Q': np.array(camintr_file['disparity_matrix']['data']).reshape(4, 4),
        'img_H': camintr_file['image_height'],
        'img_W': camintr_file['image_width'],
        'lidar2camrect': np.array(lidar2cam_file['projection_matrix']['data']).reshape(
            lidar2cam_file['projection_matrix']['rows'],
            lidar2cam_file['projection_matrix']['cols']
        ),
        'lidar2cam': np.array(lidar2cam_file['extrinsic_matrix']['data']).reshape(
            lidar2cam_file['extrinsic_matrix']['rows'],
            lidar2cam_file['extrinsic_matrix']['cols']
        )
    }

    # Optionally: save to file
    # import pickle
    # dump_calib_path = "calib_dict.pkl"
    # with open(dump_calib_path, 'wb') as f:
    #     pickle.dump(calib_dict, f)
    # import pdb; pdb.set_trace()
    return calib_dict


def compute_depth_map_single(args):
    """
    Inputs:
        root_dir: root directory of the dataset
        out_dir: output directory to save depth images
        frames_dict: all frames for a given sequence
        seq: sequence number (str)
        frame: frame index (int)
        horizon: number of frames to use (int)
        process_method: postprocessing method to use [IDW, BF]
        is_verbose: saves debug images if True
    """
    root_dir, out_dir, frames_dict, seq, frame, frame_idx, horizon, process_method, is_verbose = args
    sensors = list(frames_dict.keys())
    lidarid = sensors[0]
    cam_list = sensors[1:]

    pose_path = join(root_dir, 'poses', 'dense_global', f'{seq}.txt')
    ts_path = join(root_dir, 'timestamps', f'{seq}.txt')
    if not os.path.exists(pose_path):
        pose_path = join(root_dir, 'poses', 'dense', f'{seq}.txt')
    assert os.path.exists(pose_path), f'Pose file {pose_path} does not exist'

    pose_np = np.loadtxt(pose_path, dtype=np.float64)

    # pc_batch, accum_frames_list = load_pc_frames_pose(frames_dict[lidarid], pose_np, frame, horizon)    
    pc_batch, accum_frames_list = load_pc_frames_time(
        frames_dict[lidarid], frame_idx, horizon)
    pc_batch = transform_pc_frames(pc_batch, pose_np, frame_idx, accum_frames_list)

    depth_list = []
    if process_method == 'IDW' or process_method == "BF" or process_method == "PASS" or process_method == "STEREO":
        img_left = cv2.imread(frames_dict[cam_list[0]][frame_idx])
        img_right = cv2.imread(frames_dict[cam_list[1]][frame_idx])
        for cam in cam_list:
            calib_dict = load_calib(root_dir, seq, cam)

            is_right_cam = cam == cam_list[1]
            if is_right_cam:
                img_left, img_right = img_right, img_left
            filtered_depth = compute_stereo_depth_map(
                img_left, img_right, pc_batch, calib_dict,
                compute_right=is_right_cam,
                process_method=process_method,
                debug=is_verbose
            )
            depth_list.append(filtered_depth)
    elif process_method == 'LA' or process_method == 'LAIDW':  # Pure Lidar Accumulation
        for cam_idx, cam in enumerate(cam_list):
            img = cv2.imread(frames_dict[cam_list[cam_idx]][frame_idx])
            calib_dict = load_calib(root_dir, seq, cam)
            
            accum_depth = compute_accum_lidar_depth_map(
                img, pc_batch, calib_dict, debug=is_verbose
            )
            if process_method == 'LAIDW':
                # Add special logic for filling in depth in bottom of image
                pc_batch, accum_frames_list = load_pc_frames_time(
                    frames_dict[lidarid], frame_idx, 50)
                pc_batch = transform_pc_frames(pc_batch, pose_np, frame_idx, accum_frames_list)

                pc_accum_depth = compute_accum_lidar_depth_map(
                    img, pc_batch, calib_dict, debug=is_verbose
                )
                temp_infilldepth = compute_filter_depth_map(accum_depth)

                # For all unfilled pixels in bottom 1/4 of image, fill in with pc_accum_depth + IDW
                empty_pixels = np.where(temp_infilldepth == 0)
                top_pixels = np.where(empty_pixels[0] >= accum_depth.shape[0] * 2/3)
                empty_pixels = (empty_pixels[0][top_pixels], empty_pixels[1][top_pixels])

                pixel_mask = np.zeros(accum_depth.shape, dtype=bool)
                pixel_mask[empty_pixels] = True
                accum_depth[pixel_mask] = pc_accum_depth[pixel_mask]

                infilldepth = compute_filter_depth_map(accum_depth)
                infilldepth_mm = (infilldepth*1000).astype(np.uint16)
                accum_depth = infilldepth

            depth_list.append(accum_depth)
    else:
        raise NotImplementedError

    # 3 Save depth images
    seq_dir = join(out_dir, str(seq))
    for cam_idx, cam_name in enumerate(cam_list):
        cam_dir = join(seq_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        depth = depth_list[cam_idx]

        img_name = f'{frame}.png'
        img_path = join(cam_dir, img_name)

        # Convert depth to mm
        depth *= 1000
        depth = np.clip(depth, 0, 65535).astype(np.uint16)
        if is_verbose:
            save_depth_image(depth, img_path, colorize=True)
        else:
            save_depth_image(depth, img_path)

    return seq, frame


def main(args):

    # Parameters
    is_verbose = args.verbose
    horizon = args.scans
    process_method = args.proc
    dataset_type = args.dataset_type
    out_dir = join(
        args.out_dir, f'depth_{horizon}_{process_method}_{dataset_type}')

    if not os.path.exists(out_dir):
        print(f'Creating depth directory {out_dir}')
        os.makedirs(out_dir)

    cfg_path = args.cfg_file
    assert os.path.exists(cfg_path), f'Config file {cfg_path} does not exist'
    with open(cfg_path, 'r') as f:
        cfg_file = yaml.safe_load(f)

    # Get sensor names and their corresponding directories
    sensor_dict = [
        ["3d_raw", "os1"],
        ["2d_rect", "cam0"],
        ["2d_rect", "cam1"]
    ]

    # 0 Select Scene Type
    dataset_type_list = [ 'seq', 'all' ]
    if dataset_type in dataset_type_list:
        if dataset_type == 'all':
            input_frames_dict = get_frames_from_json(
                cfg_file['root_dir'], sensor_dict, override_all=True, ds=args.ds_factor)
        elif dataset_type == 'seq':
            input_frames_dict = {}
            for seq in args.seq_list:
                seq_input_frames_dict = get_frames_dict(
                    cfg_file['root_dir'], seq, sensor_dict, ds=args.ds_factor
                )
                # Convert to seq: {sensor: frame_path} format
                input_frames_dict.update({seq: seq_input_frames_dict})

        frames_dict = None
        total_frames = 0
        for seq, seq_frames in input_frames_dict.items():
            print(f'Processing sequence {seq}')
            total_frames += len(seq_frames['cam0'])

            # 1 Create sequence directory if it doesn't exist
            seq_dir = join(out_dir, str(seq))
            if not os.path.exists(seq_dir):
                print("Creating sequence directory", seq_dir)
                os.makedirs(seq_dir)

            # 2 Create camera directories if they don't exist
            for cam_idx, cam in enumerate(sensor_dict[1:]):
                _, sensor = cam
                cam_dir = join(seq_dir, sensor)
                os.makedirs(cam_dir, exist_ok=True)

            # 3 Iniitalize variables
            frames_dict = get_frames_dict(
                cfg_file['root_dir'], seq, sensor_dict)

            root_dir_list = [cfg_file['root_dir']] * len(seq_frames['cam0'])
            out_dir_list = [out_dir] * len(seq_frames['cam0'])
            frames_dict_list = [frames_dict] * len(seq_frames['cam0'])
            seq_list = [seq] * len(seq_frames['cam0'])
            frame_list = [
                int(path.split('/')[-1].split('_')[-1].split('.')[0])
                for path in seq_frames['cam0']
            ]
            frame_idx_list = list(range(0, args.ds_factor* len(seq_frames['cam0']), args.ds_factor))
            horizon_list = [horizon] * len(seq_frames['cam0'])
            process_method_list = [process_method] * len(seq_frames['cam0'])
            is_verbose_list = [is_verbose] * len(seq_frames['cam0'])

            # test_frame_idx = 50
            # # for test_frame_idx in range(len(seq_frames['cam0'])):
            # compute_depth_map_single((
            #     root_dir_list[test_frame_idx],
            #     out_dir_list[test_frame_idx],
            #     frames_dict_list[test_frame_idx],
            #     seq_list[test_frame_idx],
            #     frame_list[test_frame_idx],
            #     frame_idx_list[test_frame_idx],
            #     horizon_list[test_frame_idx],
            #     process_method_list[test_frame_idx],
            #     is_verbose_list[test_frame_idx]
            # ))

            # 4 Compute depth maps in parallel Uses 8 times more cores than what you specify
            COMPUTE_DEPTH_INPUTS_LIST = list(zip(
                root_dir_list,
                out_dir_list,
                frames_dict_list,
                seq_list,
                frame_list,
                frame_idx_list,
                horizon_list,
                process_method_list,
                is_verbose_list
            ))
            with Pool(processes=24) as pool:
                with tqdm(total=len(COMPUTE_DEPTH_INPUTS_LIST)) as pbar:
                    for result in pool.imap(compute_depth_map_single, COMPUTE_DEPTH_INPUTS_LIST):
                        pbar.set_description(
                            f'Processing Seq {seq} - Frame {result[1]}')
                        pbar.update()

        print(f'Finished processing {total_frames} frames')
    else:
        raise NotImplementedError

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    print("Building depth images from point clouds")
    main(args)
