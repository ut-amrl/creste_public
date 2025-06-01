import cv2
import numpy as np

import math
import torch


from torch.nn import functional as F
from creste.utils.visualization import save_depth_color_image, save_depth_image, draw_sparse_depth_on_image
from creste.utils.projection import pixels_to_depth
from creste.utils.infill import dense_map


def compute_accum_lidar_depth_map(
    img, pc_batch, calib_dict, debug=False
):
    IMG_H, IMG_W, _ = img.shape

    # 1 Accumulate point clouds
    accum_cloud = np.zeros((0, 3))
    for pc in pc_batch:
        accum_cloud = np.vstack((accum_cloud, pc[:, :3]))

    # 2 Project accumulated cloud to images
    accum_depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    valid_lidar_points, valid_lidar_depth = pixels_to_depth(
        accum_cloud[:, :3], calib_dict, IMG_H, IMG_W, IMG_DEBUG_FLAG=True
    )
    accum_depth[valid_lidar_points[:, 1],
                valid_lidar_points[:, 0]] = valid_lidar_depth

    # 3 Save accumulated depth map
    if debug:
        save_depth_color_image(img, accum_depth, 'accumdepth.png', True)
        draw_sparse_depth_on_image(
            img.copy(), valid_lidar_points, accum_depth, image_path="sparseaccumdepth.png")

    return accum_depth


def compute_filter_depth_map(depth, debug=False):
    valid_lidar_points_mask = np.logical_and(
        depth > 0, depth < 50)  # Filter out invalid depth points
    valid_depth_coords = np.array(
        list(zip(*np.nonzero(valid_lidar_points_mask))))
    valid_depth_coords = valid_depth_coords[:, ::-1]  # Convert to x, y

    H, W = depth.shape
    valid_filtered_depth = depth[valid_lidar_points_mask]
    lidar_depths = np.hstack(
        (valid_depth_coords, valid_filtered_depth.reshape(-1, 1)))
    infilldepth = dense_map(lidar_depths.T, W, H, 3)
    infilldepth[np.isnan(infilldepth)] = 0
    infilldepth[valid_depth_coords[:, 1],
                valid_depth_coords[:, 0]] = valid_filtered_depth

    if debug:
        infilldepth_mm = (infilldepth * 1000).astype(np.uint16)
        save_depth_image(infilldepth_mm, 'infilldepth.png')

    return infilldepth


def compute_stereo_depth_map(
    img1, img2, pc_batch, calib_dict,
    compute_right=False,
    process_method="IDW",
    debug=False
):
    """
    Compute dense depth map for img1 from list of accumulated point clouds and stereo depth images

    Inputs:
        img1: target image
        img2: stereo pair image
        pc_batch: list of accumulated point clouds
        calib_dict: dictionary of calibration parameters
        process_method: postprocessing method to use [IDW, BF]
    Outputs:
        depth_image: dense depth map
    """
    IMG_W, IMG_H = img1.shape[1], img1.shape[0]

    u, v = np.indices((IMG_W, IMG_H))

    depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for i in range(len(pc_batch)):
        # accum_cloud = np.vstack((accum_cloud, pc_batch[i][:, :3]))

        # 2 Project point clouds to image
        valid_lidar_points, valid_lidar_depth = pixels_to_depth(
            pc_batch[i][:, :3], calib_dict, IMG_H, IMG_W, IMG_DEBUG_FLAG=debug
        )

        # 3 Only update depth for points in unseen range
        if i > 0:
            new_depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            new_depth[valid_lidar_points[:, 1],
                      valid_lidar_points[:, 0]] = valid_lidar_depth
            # new_depth_mask = np.logical_and(new_depth > 0, new_depth < 2)
            new_depth_mask = new_depth > 0
            # update_depth_mask = np.logical_and(old_depth_mask, new_depth_mask)
            depth[new_depth_mask] = new_depth[new_depth_mask]
        else:
            depth[valid_lidar_points[:, 1],
                  valid_lidar_points[:, 0]] = valid_lidar_depth

    valid_lidar_points = np.array(list(zip(*np.nonzero(depth > 0))))
    valid_lidar_points = valid_lidar_points[:, ::-1]  # Convert to x, y

    # Draw sparse depth onto image
    clip_depth = np.clip(depth*1000, a_min=0,
                         a_max=np.iinfo(np.uint16).max)  # Convert to mm
    if debug:
        save_depth_color_image(img1, clip_depth, 'accumdepth.png', True)
        draw_sparse_depth_on_image(
            img1.copy(), valid_lidar_points, depth, image_path="sparseaccumdepth.png")
        
    # Compute stereo depth
    if compute_right:
        sgbm_depth = get_disparity(img2, img1, calib_dict, compute_right=True)
    else:
        sgbm_depth = get_disparity(img1, img2, calib_dict)

    # Compute relative error between sgbm and lidar
    abs_depth_error = np.abs(sgbm_depth - depth)
    relative_error = abs_depth_error / np.maximum(sgbm_depth, 1e-5)

    # Filter depth
    if debug:
        print("Filtering depth")
    depth_threshold = 0.3
    sgbm_mask = relative_error < depth_threshold

    valid_depth_pct = np.sum(sgbm_mask) / (IMG_H*IMG_W)
    if debug:
        print(
            f'With threshold {depth_threshold}, pct lidar pts that are valid is {valid_depth_pct}')

    filtered_depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    filtered_depth[sgbm_mask] = depth[sgbm_mask]
    if debug:
        save_depth_color_image(img1, filtered_depth, 'filtereddepth.png', True)

    if process_method == 'IDW':
        valid_lidar_points_mask = np.zeros((IMG_H, IMG_W), dtype=bool)
        valid_lidar_points_mask[valid_lidar_points[:, 1],
                                valid_lidar_points[:, 0]] = True

        valid_depth_mask = np.logical_and(valid_lidar_points_mask, sgbm_mask)
        valid_depth_coords = np.array(list(zip(*np.nonzero(valid_depth_mask))))
        valid_depth_coords = valid_depth_coords[:, ::-1]  # Convert to x, y

        valid_filtered_depth = filtered_depth[filtered_depth > 0]
        lidar_depths = np.hstack(
            (valid_depth_coords, valid_filtered_depth.reshape(-1, 1)))
        infilldepth = dense_map(lidar_depths.T, IMG_W, IMG_H, 5)
        infilldepth[np.isnan(infilldepth)] = 0
        infilldepth[valid_depth_coords[:, 1],
                    valid_depth_coords[:, 0]] = valid_filtered_depth
    elif process_method == 'BF':
        infilldepth = cv2.bilateralFilter(filtered_depth, 11, 1, 4)
    elif process_method == 'PASS':
        infilldepth = filtered_depth
    elif process_method == 'STEREO':
        infilldepth = sgbm_depth
        return infilldepth  # Return stereo depth
    else:
        raise NotImplementedError

    clip_depth = np.clip(infilldepth*1000, a_min=0,
                         a_max=np.iinfo(np.uint16).max)  # Convert to mm
    if debug:
        depth_image = save_depth_color_image(
            img1, clip_depth, 'infilldepth.png')

    # Filter depth
    if debug:
        print("Filtering infill depth")
    abs_depth_error = np.abs(sgbm_depth - infilldepth)
    relative_error = abs_depth_error / np.maximum(sgbm_depth, 1e-5)

    error_threshold = 1
    sgbm_mask = relative_error < error_threshold

    relative_error[relative_error > 255] = 0

    if debug:
        # Visualize error map
        save_depth_color_image(img1, relative_error, 'error.png')

    valid_depth_pct = np.sum(sgbm_mask) / (IMG_H*IMG_W)
    # print(f'With threshold {depth_threshold}, pct lidar pts post idw that are valid is {valid_depth_pct}')

    filtered_depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    filtered_depth[sgbm_mask] = infilldepth[sgbm_mask]
    if debug:
        save_depth_color_image(img1, filtered_depth, 'infillfiltereddepth.png')

    return filtered_depth


def get_disparity(
    image_left, image_right, left_calib_dict, compute_right=False
):
    Q = np.float32(left_calib_dict['Q']).reshape(4, 4)
    L2C = np.float32(left_calib_dict['lidar2cam']).reshape(4, 4)
    C2L = np.linalg.inv(L2C)
    H, W, C = image_left.shape
    # 1 Convert to grayscale
    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    C = 1

    # 2 Initialize SGBM settings
    numDisparities = 176  # & -16 # Make divisble by 16
    minDisparity = 0
    blockSize = 11  # Set to one to use original SGBM algorithm
    uniquenessRatio = 2
    speckleWindowSize = 50
    speckleRange = 2
    disp12MaxDiff = 1
    mode = cv2.StereoSGBM_MODE_HH  # MODE_SGBM=0. MODE_HH=1, MODE_SGBM_3WAY=2, MODE_HH4=3
    P1 = 4*C*blockSize**2  # Default 8
    P2 = 64*C*blockSize**2  # Default 32 (Must be large3r than P1)

    # 3 Compute disparity
    stereo_left = cv2.StereoSGBM.create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        preFilterCap=31,
        # uniquenessRatio = uniquenessRatio,
        # speckleWindowSize = speckleWindowSize,
        # speckleRange = speckleRange,
        # disp12MaxDiff = disp12MaxDiff,
        P1=P1,
        P2=P2,
        mode=mode
    )
    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
    disp_left = stereo_left.compute(image_left_gray, image_right_gray)
    disp_right = stereo_right.compute(image_right_gray, image_left_gray)

    # 4 Post Process disparity
    sigma = 0.8
    lmbda = 8000
    depthDiscontinuityRadius = 0  # 0 is default
    # wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(True)
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)
    # wls_filter.setDepthDiscontinuityRadius(depthDiscontinuityRadius)

    if not compute_right:
        wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(True)
        filtered_disp = wls_filter.filter(
            disp_left, image_left_gray, disparity_map_right=disp_right)
        valid_disp = filtered_disp >= 0
        save_depth_color_image(image_left, filtered_disp, 'disparity_left.png')
    else:
        wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        # wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_right)
        # Flip disparity because we are computing right disparity
        filtered_disp = wls_filter.filter(-disp_right,
                                          image_right_gray, disparity_map_right=disp_left)
        valid_disp = filtered_disp >= 0
        save_depth_color_image(
            image_right, filtered_disp, 'disparity_right.png')

    # filtered_disp = wls_filter.filter(disp_right, image_right_gray, disparity_map_right=disp_left)
    # save_depth_color_image(image_right, filtered_disp, 'disparity_right.png')
    filtered_disp = filtered_disp.astype(
        np.float32)/16.0  # Scale to 32 bit float format

    # import pdb; pdb.set_trace()
    tred_pts = cv2.reprojectImageTo3D(
        filtered_disp, Q, handleMissingValues=True)

    valid_pts_mask = np.logical_and(
        np.logical_and(tred_pts[:, :, 2] > 0, tred_pts[:, :, 2] < 60),
        np.logical_and(tred_pts[:, :, 1] < 1.2, tred_pts[:, :, 1] > -4)
    )
    depth_image = np.zeros((tred_pts.shape[0], tred_pts.shape[1]))
    depth_image[valid_pts_mask] = tred_pts[valid_pts_mask, 2]

    depth_image = np.clip(depth_image, a_min=0, a_max=np.iinfo(
        np.uint16).max)  # Convert to mm

    if not compute_right:
        save_depth_color_image(image_left, depth_image, 'colordepth.png')
        save_depth_image(depth_image, 'graydepth.png')
    else:
        save_depth_color_image(image_right, depth_image, 'colordepth.png')
        save_depth_image(depth_image, 'graydepth.png')

    return depth_image

def convert_to_metric_depth_differentiable(depth_logits, mode, depth_min, depth_max, num_bins):
    """
    Converts depth logits to metric depth in a differentiable way
    """
    depth_probs = F.softmax(depth_logits, dim=1)
    depth_bin_values = torch.linspace(
        depth_min, 
        depth_max, 
        num_bins, 
        device=depth_logits.device
    )  # [D]
    depth_bin_values = depth_bin_values.view(1, -1, 1, 1)
    depth = torch.sum(depth_probs * depth_bin_values, dim=1)
    return depth


def convert_to_metric_depth(depth_bin, mode, depth_min, depth_max, num_bins):
    """
    Converts depth bin to depth value
    Args:
        depth_bin [torch.Tensor(H, W)]: Depth bin indices
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
    Returns:
        depth [torch.Tensor(H, W)]: Depth map
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        depth = depth_bin * bin_size + depth_min
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        # TODO: Check if this is correct (might be (depth_bin)*(depth_bin+1))
        depth = depth_min + 0.5 * bin_size * (depth_bin) * (depth_bin + 1)
    elif mode == "SID":
        depth = (math.exp(math.log(1 + depth_max) - math.log(1 + depth_min)) * depth_bin / num_bins) + \
            math.log(1 + depth_min)
    else:
        raise NotImplementedError
    return depth


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * \
            torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (
            ~torch.isfinite(indices))
        indices[mask] = num_bins  # Set to last bin

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices
