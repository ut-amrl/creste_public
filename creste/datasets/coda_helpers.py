import os
import re
import glob
from os.path import join
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from creste.datasets.coda_utils import CALIBRATION_DIR
from creste.utils.projection import get_pts2pixel_transform

def get_available_sequences(indir):
    """
    Get the list of available sequences in the input directory
    """
    cam_dir = join(indir, "2d_raw", "cam0")
    if not os.path.exists(cam_dir):
        cam_dir = join(indir, "2d_rect", "cam0")
    return sorted([int(x) for x in os.listdir(cam_dir) if os.path.isdir(join(cam_dir, x))])

def load_intrinsics(indir, seq, camid):
    """
    Load the camera intrinsics from the calibration directory
    """
    calib_dir = join(indir, CALIBRATION_DIR)
    intrinsics = yaml.safe_load(
        open(join(calib_dir, str(seq), f'calib_{camid}_intrinsics.yaml'), 'r'))

    intrinsics_dict = {
        'K': np.array(intrinsics['camera_matrix']['data']).reshape(3, 3),
        'R': np.array(intrinsics['rectification_matrix']['data']).reshape(3, 3),
        'P': np.array(intrinsics['projection_matrix']['data']).reshape(
            intrinsics['projection_matrix']['rows'], intrinsics['projection_matrix']['cols']
        ),
        'img_H': intrinsics['image_height'],
        'img_W': intrinsics['image_width'],
    }
    return intrinsics_dict


def load_extrinsics(indir, seq, camid):
    """
    Load the camera to LiDAR extrinsics from the calibration directory
    """
    calib_dir = join(indir, CALIBRATION_DIR)
    extrinsics = yaml.safe_load(
        open(join(calib_dir, str(seq), f'calib_os1_to_{camid}.yaml'), 'r'))

    extrinsics_dict = {
        'lidar2cam': np.array(extrinsics['extrinsic_matrix']['data']).reshape(
            extrinsics['extrinsic_matrix']['rows'], extrinsics['extrinsic_matrix']['cols']
        ),
        'lidar2camrect': np.array(extrinsics['projection_matrix']['data']).reshape(
            extrinsics['projection_matrix']['rows'], extrinsics['projection_matrix']['cols']
        )
    }
    return extrinsics_dict


def scale_calib(calib_dict, scale):
    """
    Scales the calibration intrinsics and extrinsics by a factor
    """
    calib_dict['K'][:2, :] *= scale
    calib_dict['P'][:2, :] *= scale

    calib_dict['lidar2camrect'] = get_pts2pixel_transform(calib_dict)
    calib_dict['img_H'] = int(calib_dict['img_H'] * scale)
    calib_dict['img_W'] = int(calib_dict['img_W'] * scale)

    return calib_dict


def convert_poses_to_tf(pose_np):
    """
    Convert poses from the coda format to a 4x4 homogeneous transform.
    coda_format: ts, x, y, z, qw, qx, qy, qz
    """
    pose_quat = np.stack([pose_np[:, 5], pose_np[:, 6],
                         pose_np[:, 7], pose_np[:, 4]], axis=1)
    B = pose_np.shape[0]
    tf_pose = np.tile(np.eye(4), (B, 1, 1))
    tf_pose[:, :3, :3] = R.from_quat(pose_quat).as_matrix()
    tf_pose[:, :3, 3] = pose_np[:, 1:4]

    return tf_pose


def get_info_from_filename(path, ext='bin'):
    """
    Extracts the sequence identifier and frame number from a filename that can vary in format.
    The function is designed to handle both filenames that end with a frame number like '6438.bin'
    and more complex paths that include a sequence followed by a frame number.

    Args:
        path (str): The full path of the file, expected to end with '_<frame_number>.ext' or just '<frame_number>.ext'.
        ext (str): The file extension without the period, typically 'bin'.

    Returns:
        str or None: A string with the sequence identifier (if applicable) and frame number if found, otherwise None.
    # """
    # pattern = r'.*/([^/]+)/[^/]*?(\d+)\.' + re.escape(ext) + r'$'
    # match = re.search(pattern, path)
    # if match:
    #     sequence = match.group(1)  # Parent directory name as the sequence
    #     frame = match.group(2)      # Frame number
    #     return f'{sequence} {frame}'

    # return None
    # Regex to find the last numeric part before the extension, considered as the frame number
    pattern = r'(.*)/([^/]*?)(\d+)\.' + re.escape(ext) + r'$'
    match = re.search(pattern, path)
    if match:
        # Everything before the frame number as part of the sequence
        pre_path = match.group(1)
        potential_sequence = match.group(2).strip('_')
        frame = match.group(3)

        # Use all non-numeric segments in pre_path and potential_sequence as the sequence identifier
        sequence_parts = re.split(r'/+', pre_path) + [potential_sequence]
        for part in sequence_parts[::-1]:
            if part.isdigit():
                sequence = part
                break

        return f'{sequence} {frame}'

    return None


def sort_func(x, ext="bin"):
    """Creates a naturally sorted key from seq and frame"""
    seq, frame = get_info_from_filename(x, ext=ext).split(' ')

    return int(seq) * 100000 + int(frame)


def pose_sort_func(x):
    """Extracts the seq number from pose filepath"""
    seq = os.path.basename(x).split('.txt')[0]
    return int(seq)


def get_dir_frame_info(seq_dir, ext="bin", short=False):
    if 'cam0' in os.listdir(seq_dir):
        seq_dir = join(seq_dir, 'cam0')

    """Loads all frames for a given sequence"""
    infos_files = sorted(glob.glob(join(seq_dir, f'*.{ext}')),
                         key=lambda x: sort_func(x, ext)
                         )

    if short:
        infos_files = [get_info_from_filename(f, ext=ext) for f in infos_files]
    return infos_files

# def get_dir_frame_info(seq_dir: str, ext: str = "bin", short: bool = False):
#     """
#     Collect every file with the given extension under `seq_dir`, *recursively*.

#     Parameters
#     ----------
#     seq_dir : str
#         Root directory of the sequence (will be walked recursively).
#     ext : str, default "bin"
#         File extension to match (without leading dot).
#     short : bool, default False
#         If True, return a list of objects produced by `get_info_from_filename`
#         instead of the raw paths.

#     Returns
#     -------
#     list
#         Either full paths to each matching file (sorted), or—if `short` is
#         True—the processed info objects in the same order.
#     """
#     # ** matches any depth of sub-directories when recursive=True
#     pattern = join(seq_dir, "**", f"*.{ext}")
#     infos_files = sorted(
#         glob.glob(pattern, recursive=True),
#         key=lambda x: sort_func(x, ext)      # keep existing sort order
#     )

#     if short:
#         infos_files = [get_info_from_filename(f, ext=ext) for f in infos_files]

#     return infos_files


def get_sorted_subdirs(root_dir, exclude_dirs=[]):
    """Loads all sequence directories for a given directory"""
    seq_subdirs = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)) and len(d) < 3], key=int
                         )
    seq_subdirs = [join(root_dir, d)
                   for d in seq_subdirs if int(d) not in exclude_dirs]
    return seq_subdirs


def get_bev_patch_from_pose(pose_np):
    """
    Input: pose_np [Nx4x4] poses in the global frame
    Output: bev_patch [NxPx2] patches in the global frame, where P is the max number of patches and 
                    2 is the u, v coordinates of the patch in the global BEV frame
    """
    # Infer the min and max values of the map
    fov = 70
    # num_samples = 20
    As = 10
    Ds = 10
    pad_x, pad_y, = 20, 20
    view_distance = 12.8
    min_x = pose_np[:, 0, 3].min() - pad_x
    min_y = pose_np[:, 1, 3].min() - pad_y
    max_x = pose_np[:, 0, 3].max() + pad_x
    max_y = pose_np[:, 1, 3].max() + pad_y

    map_min = np.array([min_x, min_y], dtype=np.float32)
    map_max = np.array([max_x, max_y], dtype=np.float32)
    map_res = np.array([0.1, 0.1], dtype=np.float32)

    # Define the patch size
    grid_min = np.array([0, 0], dtype=np.int32)
    grid_max = ((map_max - map_min) / map_res).astype(np.int32)

    # Sample along x and y bounds of each pose to the patch samples
    position = pose_np[:, :2, 3]
    rot_angle = np.arctan2(pose_np[:, 1, 0], pose_np[:, 0, 0])[..., None]
    B = pose_np.shape[0]

    # Generate horizontal FOV angles
    dtheta = np.radians(np.linspace(-fov / 2, fov / 2, As))
    ddist = np.linspace(0.2, view_distance, Ds)
    bev_patches = np.zeros((pose_np.shape[0], As*Ds, 2))

    # Calculate the ray endpoints
    total_angle = rot_angle + dtheta  # theta + dtheta
    total_angle = total_angle.reshape(B*As,)
    ray_dir = np.stack(
        [np.cos(total_angle), np.sin(total_angle)], axis=-1
    )  # [B*N, 2]
    # [B*As, 2, 1] @ [1, Ds] -> [B*As, 2, Ds]
    dpos = ray_dir.reshape(B*As, 2, 1) @ ddist.reshape(1, -1)
    dpos = np.swapaxes(dpos, 1, 2)  # [B*N, 2, Ds] -> [B*N, Ds, 2]

    # Convert to bev patch index
    position_rep = np.repeat(position, As, axis=0)[
        :, None, :]  # [B, 1, 2] -> [B*As, 1, 2]
    # [B*As, Ds, 2] -> [B, As, Ds, 2]
    endpoints = (position_rep + dpos).reshape(B, As, Ds, 2)
    # [B, As, Ds, 2] -> [B, As*Ds, 2]
    endpoints = endpoints.reshape(B, As*Ds, 2)
    bev_patches = (endpoints - map_min) / map_res  # [B, As*Ds, 2]
    bev_patches_dis = np.clip(bev_patches, grid_min, grid_max).astype(np.int32)

    # Debug only visualize the patches
    if True:
        import matplotlib.pyplot as plt
        import cv2
        testimg = np.zeros(
            (int(grid_max[1]), int(grid_max[0]), 3), dtype=np.uint8)

        plt.figure()
        # Randomly sample patches at fixes lengths apart
        color_list = ['r', 'g', 'b']
        color_idx = 0
        for frame in range(0, B, 100):
            color = color_list[color_idx % 3]
            color_idx += 1
            plt.scatter(
                endpoints[frame, :, 0],
                endpoints[frame, :, 1],
                c=color
            )
            for j in range(As*Ds):
                u, v = bev_patches_dis[frame, j]
                testimg[v, u] = [255, 255, 255]

            # if color_idx > 2:
            #     break
        plt.title("Sampled FOV patches from robot poses")
        plt.savefig('bev_patches.png')
        plt.close()

        # testimg = np.zeros((int((max_y - min_y) / grid_res[1]), int((max_x - min_x) / grid_res[0]), 3), dtype=np.uint8)
        # for j in range(As*Ds):
        #     u, v = bev_patches_dis[0, j]
        #     testimg[v, u] = [255, 255, 255]
        cv2.imwrite('bev_patches_dis.png', testimg)
    import pdb
    pdb.set_trace()
    return bev_patches_dis
