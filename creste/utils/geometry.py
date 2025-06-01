import math
import numpy as np
from shapely.geometry import Point, Polygon

def sector(center, start_angle, end_angle, radius, steps=200):
    def polar_point(origin_point, angle,  distance):
        return [origin_point.x + math.sin(math.radians(angle)) * distance, origin_point.y + math.cos(math.radians(angle)) * distance]

    if start_angle > end_angle:
        start_angle = start_angle - 360
    else:
        pass
    step_angle_width = (end_angle-start_angle) / steps
    sector_width = (end_angle-start_angle) 
    segment_vertices = []

    segment_vertices.append(polar_point(center, 0,0))
    segment_vertices.append(polar_point(center, start_angle,radius))

    for z in range(1, steps):
        segment_vertices.append((polar_point(center, start_angle + z * step_angle_width,radius)))
    segment_vertices.append(polar_point(center, start_angle+sector_width,radius))
    segment_vertices.append(polar_point(center, 0,0))
    return Polygon(segment_vertices)

def get_overlapping_views(
    query_pose_idx, db_poses_se3, tp_min=0.1, tp_max=0.8, fov=70, view_dist=12.8, max_dist=19.2):
    """
    Computes intersection between camera views from the query pose and 
    db_poses. Uses shapely to compute the area between two intersections 
    and returns the area of intersection. Does this in two steps:
    Source: https://stackoverflow.com/questions/54284984/sectors-representing-and-intersections-in-shapely

    Step 1: Coarse check, computes the euclidean distance between camera poses, if distance exceeds camera fov threshold, marks views as non-overlapping
    Step 2: Fine check, computes the intersection area between camera view sectors. If more than 50% of the area is overlapping, marks views as overlapping
    
    """
    def pose2sector(pose, fov, distance):
        angle = np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))
        return sector(
            Point(pose[0, 2], pose[1, 2]), 
            angle - fov/2, angle + fov/2, 
            distance
        )

    # import time
    # start = time.time()
    B = db_poses_se3.shape[0]
    db_pose = np.zeros((B, 3, 3))
    db_pose[:, :2, :2] = db_poses_se3[:, :2, :2]
    db_pose[:, :2, 2] = db_poses_se3[:, :2, 3]

    query_pose = db_pose[query_pose_idx]

    # # Sanity check for poses
    # import pdb; pdb.set_trace()
    # test_db_sh_list = [pose2sector(pose, fov, view_dist) for pose in db_pose[:1000]]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for id in range(len(test_db_sh_list)):
    #     plt.plot(*test_db_sh_list[id].exterior.xy)
    # import pdb; pdb.set_trace()

    # Step 1: Coarse check
    pose_dist = np.linalg.norm(
        db_pose[:, :2, 2] - query_pose[:2, 2], axis=1
    )
    coarse_overlap = pose_dist < max_dist

    db_coarse = db_pose[coarse_overlap] 
    qp_sh = pose2sector(query_pose, fov, view_dist)
    db_sh_list = [pose2sector(pose, fov, view_dist) for pose in db_coarse]

    # Step 2: Fine check
    intersection_area = np.array([
        qp_sh.intersection(db_sh).area
        for db_sh in db_sh_list
    ], dtype=np.float32)
    
    # Compute area of query sector
    max_overlap_area = qp_sh.area
    # fine_overlap = np.logical_and(
    #     (intersection_area / max_overlap_area) > tp_min,
    #     (intersection_area / max_overlap_area) < tp_max
    # )
    fine_overlap = ((intersection_area / max_overlap_area) > tp_min) & ((intersection_area / max_overlap_area) < tp_max)

    # Select ids of overlapping poses
    overlap = np.zeros((B,), dtype=np.float32)
    overlap[coarse_overlap] = fine_overlap
    overlap_ids = np.where(overlap)[0]

    fine_overlap_ratio = intersection_area[fine_overlap] / max_overlap_area
    # overlap_infos = np.stack([overlap_ids, fine_overlap_ratio], axis=1).astype(np.float32)
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(*qp_sh.exterior.xy)
    # for id in overlap_ids:
    #     plt.plot(*db_sh_list[id].exterior.xy)
    # import pdb; pdb.set_trace()
    # Combine coarse and fine checks
    # overlap = np.zeros((B, 2), dtype=np.float32)

    # overlap[coarse_overlap] = fine_overlap
    # overlap[query_pose_idx] = False # Exclude the query pose
    # print("Time taken: ", time.time() - start, "s")
    return {"overlap_ids": overlap_ids.astype(np.int32), "overlap_ratio": fine_overlap_ratio}

def transform_poses(poses, ref_idx=0):
    """
    Computes transformation of poses from global frame to current frame
    Inputs:
        poses - list of poses in global frame  (4x4 numpy array)
    Outpus:
        rel_poses - list of poses in current frame (4x4 numpy array)
    """

    #1 Compute relative poses
    T_ref_global = poses[ref_idx]
    pose_horizon = np.empty((0, 4, 4), dtype=np.float32)
    for pose in poses:
        T_global = pose
        T_rel = np.linalg.inv(T_ref_global) @ T_global
        pose_horizon = np.concatenate((pose_horizon, T_rel[None,...]), axis=0)

    return pose_horizon

def get_2d_direction_and_position(pose):
    """ Extract the 2D position and forward direction vector from a 3x4 SE3 pose matrix. """
    position = pose[:2, 3]
    forward_vector = pose[:2, 2]  # Assuming z-axis is the forward direction vector
    forward_vector /= np.linalg.norm(forward_vector)  # Normalize the vector
    return position, forward_vector

def vectorized_fov_overlap_2d(pose1, poses2, fov_angle_deg, range_radius):
    """
    Checks for FOV overlap between one pose and multiple other poses in bird's eye view.

    Args:
        pose1 (numpy.ndarray): The SE3 pose of the reference camera (3x4 submatrix).
        poses2 (numpy.ndarray): An array of SE3 poses for the other cameras (Nx3x4).
        fov_angle_deg (float): The FOV angle in degrees.
        range_radius (float): The effective range radius of the cameras in the same units as positions.

    Returns:
        numpy.ndarray: A boolean array where each element is True if there is an overlap with pose1, False otherwise.
    """
    pos1, dir1 = get_2d_direction_and_position(pose1)
    positions, directions = get_2d_direction_and_position(poses2.transpose((1, 2, 0)))
    positions = positions.transpose()
    directions = directions.transpose()

    # Convert degrees to radians for calculations
    fov_angle_rad = np.radians(fov_angle_deg / 2)  # Half angle

    # Calculate angles for all directions
    angle1 = np.arctan2(dir1[1], dir1[0])
    angles2 = np.arctan2(directions[:, 1], directions[:, 0])

    # Calculate angle differences
    diff_angles = np.abs(angle1 - angles2) % (2 * np.pi)
    diff_angles = np.minimum(diff_angles, 2 * np.pi - diff_angles)  # Normalize to smallest angle

    # Check if angles are within the FOV
    angle_checks = diff_angles <= fov_angle_rad

    # Calculate distances
    distances = np.linalg.norm(positions - pos1, axis=1)

    # Check if distances are within range
    distance_checks = distances <= range_radius * 2

    # Combine checks
    overlap_checks = angle_checks & distance_checks

    return overlap_checks
