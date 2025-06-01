import torch_scatter
import torch
import cv2
import numpy as np

import os
import sys

from creste.utils.visualization import numpy_to_pcd

def get_pixel2pts_transform(calib_dict):
    """
    Returns a transformation matrix that converts image pixels to 3D points in LiDAR frame

    Inputs:
        calib_dict: [dict] calibration dictionary
    Outputs:
        pix2pts: [4 x 4] transformation matrix
    """
    # Assume finite camera eqn
    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :] = calib_dict['lidar2cam'][:3, :]
    T_cam_lidar = np.linalg.inv(T_lidar_cam)  # 4x4

    T_canon = np.eye(4)
    T_canon[:3, :3] = calib_dict['R'].T  # 4x4

    M = calib_dict['P'][:3, :3]
    P_pix_cam = np.eye(4)
    P_pix_cam[:3, :3] = np.linalg.inv(M)  # 4x4

    T_rect_to_lidar = T_cam_lidar @ T_canon @ P_pix_cam

    return T_rect_to_lidar


def get_pts2pixel_transform(calib_dict):
    """
    Returns a transformation matrix that converts 3D points in LiDAR frame to image pixel coordinates
    Boilerplate function to get the projection matrix from the calibration dictionary

    P =  Pcam @ Eye(Re | 0) @ T_lidar_cam

    Inputs:
        calib_dict: [dict] calibration dictionary
    Outputs:
        pts2pix: [4 x 4] transformation matrix
    """
    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :] = calib_dict['lidar2cam'][:3, :]

    T_canon = np.eye(4)
    T_canon[:3, :3] = calib_dict['R']

    M = calib_dict['P'][:3, :3]
    P_pix_cam = np.eye(4)
    P_pix_cam[:3, :3] = M

    T_lidar_to_rect = P_pix_cam @ T_canon @ T_lidar_cam

    return T_lidar_to_rect


def pixels_to_depth(
        pc_np, calib, IMG_H, IMG_W, return_keys=['image_pts', 'image_depth'], IMG_DEBUG_FLAG=False, depth_priority="max"):
    """
    pc_np:      [N x >=3] point cloud in LiDAR frame
    image_pts   [N x uv]
    calib:      [dict] calibration dictionary
    IMG_W:      [int] image width
    IMG_H:      [int] image height
    return_keys: [list] keys to return ['image_pts', 'image_depth']
    IMG_DEBUG_FLAG: [bool] flag to save debug images
    depth_priority: [str] method to prioritize depth values

    Returns depth values in meters
    """
    # Convert to numpy if tensor
    if isinstance(pc_np, torch.Tensor):
        pc_np = pc_np.cpu().numpy()

    lidar2camrect = calib['lidar2camrect']
    if isinstance(lidar2camrect, torch.Tensor):
        lidar2camrect = lidar2camrect.cpu().numpy()

    # Remove points behind camera after coordinate system change
    # Remove intensity and scale for opencv
    pc_np = pc_np[:, :3].astype(np.float64)
    pc_homo = np.hstack((pc_np, np.ones((pc_np.shape[0], 1))))
    pc_rect_cam = (lidar2camrect @ pc_homo.T).T

    # [Nx4] -> [Nx3] in the event we us auxiliary dimension to pose transform
    pc_rect_cam = pc_rect_cam[:, :3]

    lidar_pts = pc_rect_cam / pc_rect_cam[:, -1].reshape(-1, 1)
    MAX_INT32 = np.iinfo(np.int32).max
    MIN_INT32 = np.iinfo(np.int32).min
    lidar_pts = np.clip(lidar_pts, MIN_INT32, MAX_INT32)
    lidar_pts = lidar_pts.astype(np.int32)[:, :2]

    pts_mask = pc_rect_cam[:, 2] > 0

    in_bounds = np.logical_and(
        np.logical_and(lidar_pts[:, 0] >= 0, lidar_pts[:, 0] < IMG_W),
        np.logical_and(lidar_pts[:, 1] >= 0, lidar_pts[:, 1] < IMG_H)
    )

    valid_point_mask = in_bounds & pts_mask
    valid_lidar_points = lidar_pts[valid_point_mask, :]
    valid_lidar_depth = pc_rect_cam[valid_point_mask, 2]  # Use z in cam frame

    if IMG_DEBUG_FLAG:
        test_img = np.zeros((IMG_H, IMG_W), dtype=int)
        test_img[valid_lidar_points[:, 1], valid_lidar_points[:, 0]] = 255
        cv2.imwrite("test.png", test_img)

    # 1 Create LiDAR depth image
    depth_image_np = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    depth_image_np[valid_lidar_points[:, 1],
                   valid_lidar_points[:, 0]] = valid_lidar_depth

    # 2 Select farthest point in pixel to use as depth
    loc1d = valid_lidar_points[:, 1] * IMG_W + valid_lidar_points[:, 0]
    max_depth_th = torch_scatter.scatter(torch.tensor(valid_lidar_depth),
                                         torch.tensor(loc1d, dtype=torch.long),
                                         dim=0,
                                         dim_size=IMG_H*IMG_W,
                                         reduce=depth_priority)
    max_depth = max_depth_th.cpu().numpy()
    max_depth = max_depth.reshape((IMG_H, IMG_W))
    valid_depth_points = np.array(list(zip(*np.nonzero(max_depth))))
    valid_depth_points[:, [0, 1]
                       ] = valid_depth_points[:, [1, 0]]  # Swap x and y
    valid_depth = max_depth[max_depth != 0].reshape(-1,)

    if IMG_DEBUG_FLAG:
        depth_mm = (max_depth * 1000).astype(np.uint16)
        cv2.imwrite("pp_depth_max.png", depth_mm)

    return_values = []
    for key in return_keys:
        if key == 'image_pts':
            return_values.append(valid_depth_points)
        elif key == 'image_depth':
            return_values.append(valid_depth)
        elif key == 'depth':
            return_values.append(depth_image_np)
        elif key == 'pc_pts':
            return_values.append(valid_lidar_points)
        elif key == 'pc_mask':
            return_values.append(valid_point_mask)
        else:
            raise ValueError(f"Invalid key {key} in return_keys")

    return return_values

# Helpers functions for converting from camera to xyz coordinate frames


def cam2world(inputs):
    """
    Inputs:
        rgbd: torch.Tensor(B, 1, H, W)
        p2p: torch.Tensor(B, 4, 4)
    Ouputs:
        xyz: torch.Tensor(B, 3, H, W)
    """
    depth, p2p, bev_params = inputs
    B, _, H, W = depth.shape

    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    campts = torch.tile(
        torch.stack([
            u, v, torch.ones_like(u)
        ], dim=0), (B, 1, 1, 1)
    ).to(depth.device)
    campts = campts * depth
    campts = torch.cat([campts, torch.ones_like(depth)], dim=1)
    _, _, H, W = campts.shape

    xyz = torch.bmm(p2p, campts.flatten(start_dim=2))
    xyz = xyz.view(B, 4, H, W)[:, :3, :, :]  # [B, C, H, W]
    xyz = xyz.permute(0, 2, 3, 1).view(
        B, H*W, -1)  # [B, C, H, W] -> [B, H*W, C]

    xyz_mask = torch.all(
        (xyz < bev_params['max_bound']) & (xyz >= bev_params['min_bound']),
        dim=2, keepdim=True
    )
    xyz_mask = xyz_mask.view(B, H, W, 1).permute(
        0, 3, 1, 2)  # [B, H, W, 1] -> [B, 1, H, W]
    xyz = xyz.view(B, H, W, -1).permute(0, 3, 1, 2)

    return xyz, xyz_mask

# Helper functions for converting from xyz to voxel coordinates


def points2voxels(inputs):
    """
    Inputs:
        points: torch.Tensor(B, HW, 3)
        bev_params: Dict of bev map params
    Outputs:
        voxels: torch.Tensor(B, HW, 2)
    """
    points, bev_params = inputs
    points = torch.cat([points, torch.ones_like(points[:, :, :1])], dim=2)
    points = (bev_params['lidar2map'] @
              points.permute(0, 2, 1)).permute(0, 2, 1)
    voxels = points[:, :, :2] / bev_params['voxel_size'][:2]

    # Clip to map size
    min_bound = torch.zeros_like(bev_params['map_size'][:2])
    voxels = torch.clamp(voxels, min_bound, bev_params['map_size'][:2] - 1)

    return voxels.long()


if __name__ == "__main__":
    import yaml
    import numpy as np
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    seq = 6
    frame = 1900
    cam_ids = ["cam0", "cam1"]
    for cam_id in cam_ids:
        camintr_path = f'/robodata/arthurz/Research/CompleteNet/data/coda/calibrations/{seq}/calib_{cam_id}_intrinsics.yaml'
        lidar2cam_path = f'/robodata/arthurz/Research/CompleteNet/data/coda/calibrations/{seq}/calib_os1_to_{cam_id}.yaml'
        camintr_file = yaml.safe_load(open(camintr_path, 'r'))
        lidar2cam_file = yaml.safe_load(open(lidar2cam_path, 'r'))

        calib_dict = {
            'K': np.array(camintr_file['camera_matrix']['data']).reshape(3, 3),
            'R': np.array(camintr_file['rectification_matrix']['data']).reshape(3, 3),
            'P': np.array(camintr_file['projection_matrix']['data']).reshape(
                camintr_file['projection_matrix']['rows'], camintr_file['projection_matrix']['cols']
            ),
            'lidar2cam': np.array(lidar2cam_file['extrinsic_matrix']['data']).reshape(
                lidar2cam_file['extrinsic_matrix']['rows'], lidar2cam_file['extrinsic_matrix']['cols']
            )
        }
        p2p = get_pixel2pts_transform(calib_dict)

        # Load depth image
        depthimg_path = f'/robodata/arthurz/Research/CompleteNet/data/coda/depth_50_LA_semistatic/{seq}/{cam_id}/{frame}.png'
        depth = cv2.imread(depthimg_path, -1).astype(np.float32) / 1000.0
        # pc_path = f'/robodata/arthurz/Research/CompleteNet/data/coda/3d_comp/os1/{seq}/3d_comp_os1_{seq}_{frame}.bin'
        # pc_np = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        # pc_np = pc_np[:, :3] # Remove intensity

        # T_lidar_cam = calib_dict['lidar2cam']
        # T_cam_camrect = np.zeros((3, 4))
        # T_cam_camrect[:3, :3] = calib_dict['R']
        # M = calib_dict['P'][:3, :3]
        # T_camrect_pixels = M
        # T_lidar_pixels = T_camrect_pixels @ T_cam_camrect @ T_lidar_cam
        # calib_dict['lidar2camrect'] = T_lidar_pixels

        # print(f'OS1 to {cam_id}:\n', calib_dict['lidar2camrect'])

        # # Compute depth image from LiDAR point cloud
        # H, W = camintr_file['image_height'], camintr_file['image_width']
        # depth = pixels_to_depth(pc_np, calib_dict, H, W, return_keys=["depth"], IMG_DEBUG_FLAG=True)

        from models.blocks.splat_projection import Camera2World
        cam2world = Camera2World()

        # Convert depth image to point cloud
        depth = torch.tensor(
            depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        p2p = torch.tensor(p2p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        xyz = cam2world((depth, p2p)).squeeze().permute(1, 2, 0)  # HxWx3

        xyz = xyz.numpy().reshape(-1, 3)
        depth_mask = xyz[:, 2] > 0
        xyz = xyz[depth_mask, :]

        # Save pointcloud to visualize
        pcd_file = f'projection_{cam_id}.pcd'
        numpy_to_pcd(xyz, pcd_file)
        print(f'Saved {pcd_file}!')
