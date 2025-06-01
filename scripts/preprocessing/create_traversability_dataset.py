import os
import glob
from os.path import join
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

from creste.datasets.coda_utils import TRAVERSE_LABEL_DIR, TRAVERSE_LABEL_SUBDIR, POSES_DIR, POSES_SUBDIRS
import creste.datasets.coda_helpers as ch

def parse_args():
    parser = argparse.ArgumentParser(description='Build traversability dataset')
    parser.add_argument('--indir', type=str, default='./data/creste', help="Path to root directory")
    parser.add_argument('--outdir', type=str, default='./data/creste/traversability', help="Path to output directory")
    parser.add_argument('--skip_factor', type=int, default=5, help='Number of poses to skip between frames')
    parser.add_argument('--dist_thresh', type=float, default=2, help='Minimum distance traversed between start to finish (m)')
    parser.add_argument('--speed_thresh', type=float, default=0.1, help='Minimum initial speed (m/s)')
    parser.add_argument('--num_frames', type=float, default=50, help='Number of future frames per pose')
    args = parser.parse_args()
    return args

def quaternion_to_forward_vector(quaternion):
    """
    Converts a quaternion to a forward direction vector.
    
    Inputs:
        quaternion: array-like, shape (4,) - The quaternion [qx, qy, qz, qw].
    
    Outputs:
        forward_vector: numpy array, shape (3,) - The forward direction vector.
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    
    # The forward vector in the local frame is along the z-axis [0, 0, 1]
    forward_vector = rotation_matrix @ np.array([0, 0, 1])
    
    return forward_vector

def parse_sequence(poses_np, skip_factor, dist_thresh, speed_thresh, num_frames):
    """
    Filters the sequence of poses to only include those where in the future frames 
    the robot moves a minimum distance between start to finish, has a minimum initial speed, and
    all the poses are in the field of view mask.
    Inputs:
        poses_np: numpy array of shape [N, 8] where N is the number of poses and 8 is the pose vector [ts,x, y, z, qw, qx, qy, qz]    
        skip_factor: number of poses to skip between frames
        dist_thresh: minimum distance traversed between start to finish (m)
        speed_thresh: minimum initial speed (m/s)
        num_frames: number of future frames per pose
    Outputs:
        pose_ids_np: numpy array of shape [M, 2] where M are the indices of poses that match the criteria
    """
    # Calculate the indices for start and end poses
    start_indices = np.arange(0, len(poses_np) - num_frames, skip_factor)
    end_indices = start_indices + num_frames

    # Extract start and future poses
    start_poses = poses_np[start_indices, :]
    future_poses = poses_np[end_indices, :]
    
    # Calculate distances between start and future poses, only compare x and y
    dist_traveled = np.linalg.norm(future_poses[:, 1:3] - start_poses[:, 1:3], axis=1)

    # Extract quaternions, flip the order to scalar last
    start_quaternions = start_poses[:, 4:8]
    start_quaternions = np.column_stack((start_quaternions[:, 1:], start_quaternions[:, 0]))
    future_quaternions = future_poses[:, 4:8]
    future_quaternions = np.column_stack((future_quaternions[:, 1:], future_quaternions[:, 0]))
    
    # Convert quaternions to rotation objects
    start_rotations = R.from_quat(start_quaternions)
    future_rotations = R.from_quat(future_quaternions)
    
    # Transform future quaternions to be relative to start quaternions
    relative_rotations = start_rotations.inv() * future_rotations
    
    # Calculate the forward direction vector for the relative rotations
    relative_forward_vectors = relative_rotations.apply([0, 0, 1])
    
    # We only care about the rotation in the xy-plane (z-axis)
    forward_alignment = relative_forward_vectors[:, 2]
    # # Calculate time differences
    # time_diff = future_poses[:, 0] - start_poses[:, 0]
    
    # # Avoid division by zero
    # valid_time_diff = time_diff > 0
    
    # # Calculate initial speed
    # initial_speed = np.zeros_like(dist_traveled)
    # initial_speed[valid_time_diff] = dist_traveled[valid_time_diff] / time_diff[valid_time_diff]
    # Apply thresholds
    valid_mask = np.logical_and(dist_traveled >= dist_thresh, forward_alignment > 0)
    valid_indices = start_indices[valid_mask]

    return valid_indices

def main(args):
    print(args)

    print("Building traversability dataset")
    # Load the poses
    poses_dir = join(args.indir, POSES_DIR, POSES_SUBDIRS[0])
    if not os.path.exists(poses_dir):
        poses_dir = join(args.indir, POSES_DIR, POSES_SUBDIRS[1])
    assert os.path.exists(poses_dir), f"Poses directory {poses_dir} does not exist"
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pose_subpaths = [f for f in 
        sorted(glob.glob(join(poses_dir, f'*.txt')), key=lambda x: ch.pose_sort_func(x))]
    poses_np = np.empty((0, 8), dtype=np.float64)
    for pose_subpath in pose_subpaths:
        seq = ch.pose_sort_func(pose_subpath)
        poses_np = np.loadtxt(pose_subpath, dtype=np.float64)
        valid_pose_frames = parse_sequence(poses_np, args.skip_factor, args.dist_thresh, args.speed_thresh, args.num_frames)

        # Create np array of the seq and frame indices of the valid poses
        pose_ids_np = np.column_stack((np.ones_like(valid_pose_frames) * seq, valid_pose_frames))

        # Save the valid pose frames
        outpath = join(outdir, f'{seq}.txt')
        print("Saving to", outpath)
        np.savetxt(outpath, pose_ids_np, fmt='%d')
    print("Done")

if __name__ == "__main__":
    args = parse_args()
    main(args)