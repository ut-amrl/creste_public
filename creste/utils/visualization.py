import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
import colorsys
import gc
import torch_scatter
import kornia

from creste.datasets.coda_utils import REMAP_SEM_ID_TO_COLOR, SEM_ID_TO_COLOR, REMAP_OBJ_ID_TO_COLOR, OBJ_ID_TO_COLOR, SSC_LABEL_DIR, SOC_LABEL_DIR, FSC_LABEL_DIR, ELEVATION_LABEL_DIR, SAM_LABEL_DIR, TRAVERSE_LABEL_DIR, SAM_DYNAMIC_LABEL_DIR, SAM_DYNAMIC_COLOR_MAP
from PIL import Image

import vispy
try:
    vispy.use('egl')
except:
    print("EGL not available, using default backend")
import vispy.scene
from vispy.scene import visuals
import creste.utils.visualize_elevation_gt as vegt
import creste.utils.feature_extractor as fe

from creste.utils.utils import make_labels_contiguous_vectorized


def resize_and_pad_image(image, max_height, max_width):
    """
    Resize an image to fill the max_height and max_width as much as possible,
    and pad the rest with black pixels.

    Parameters:
        image (numpy.ndarray): The input image to resize and pad.
        max_height (int): The maximum height of the output image.
        max_width (int): The maximum width of the output image.

    Returns:
        numpy.ndarray: The resized and padded image.
    """
    # Original dimensions
    original_height, original_width = image.shape[:2]

    # Determine the new size, preserving the aspect ratio
    ratio_height = max_height / original_height
    ratio_width = max_width / original_width
    ratio = min(ratio_height, ratio_width)

    new_height = int(original_height * ratio)
    new_width = int(original_width * ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Prepare to pad
    top_pad = (max_height - new_height) // 2
    bottom_pad = max_height - new_height - top_pad
    left_pad = (max_width - new_width) // 2
    right_pad = max_width - new_width - left_pad

    # Pad the resized image
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image


def save_preds_image(
        image_path,
        preds,
        labels,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.5,
        color=(255, 255, 255),
        thickness=1,
        verbose=True
):
    """
    Inputs:
        image_path  - path to save image
        preds       - [H, W] predicted labels
        gt_labels   - [H, W] ground truth labels
    """
    assert labels.shape == preds.shape, \
        f'gt_labels.shape {labels.shape} != preds.shape {preds.shape}'

    H, W = preds.shape
    # Convert to RGB
    id2rgb = np.array(REMAP_SEM_ID_TO_COLOR, dtype=np.uint8)
    pred_rgb_map = id2rgb[preds]
    gt_rgb_map = id2rgb[labels]

    # Save image
    pred_text_location = (10, 15)
    gt_text_location = (2*W - 20, 15)
    img = np.concatenate((pred_rgb_map, gt_rgb_map), axis=1)
    img = cv2.putText(
        img, 'Pred', pred_text_location,
        font, font_scale, color, thickness, cv2.LINE_AA
    )
    img = cv2.putText(
        img, 'GT', gt_text_location,
        font, font_scale, color, thickness, cv2.LINE_AA
    )

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, img)
    if verbose:
        print('Saved image to', image_path)


def save_depth_image(depth, img_path, colorize=False):
    """
    Assumes depth is in mm
    """
    # norm_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # norm_depth = cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16UC1)
    # norm_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)

    if colorize:
        depth = depth.clip(0, 12800)
        norm_depth = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        norm_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)
    else:
        norm_depth = depth.astype(np.uint16)
    cv2.imwrite(img_path, norm_depth)

    return norm_depth


def save_depth_color_image(rgb, depth, img_path, debug=False):
    """
    Assumes depth is in meters

    Inputs:
        rgb - [H, W, 3] np.ndarray BGR Format
        depth - [H, W] np.ndarray in meters (mm)
        img_path - path to save image
    Outputs:
        None
    """
    depth = depth.clip(0, 12.8)
    norm_rgb = cv2.normalize(
        rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    norm_depth = cv2.normalize(
        depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # norm_depth = cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16UC1)
    norm_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)

    if debug:
        print("Saving depth color image to", img_path)

    alpha = 0.2
    rgbd_image = norm_rgb.copy()
    cv2.addWeighted(rgbd_image, alpha, norm_depth, 1-alpha, 0, rgbd_image)
    cv2.imwrite(img_path, rgbd_image)

    return norm_depth


def draw_sparse_depth_on_image(
        image,
        pts,
        depth,
        radius=1,
        thickness=1,
        verbose=True,
        image_path=""
):
    """
    Inputs:
        image - np image array [H, W, 3]
        pts - [N, 2] pixel coordinates
        depth - [H, W] depth image
    """
    # import pdb; pdb.set_trace()
    depth = depth.clip(0, 12.8)
    norm_depth = cv2.normalize(
        depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)

    # Load image
    for pt_idx in range(len(pts)):
        pt = pts[pt_idx].astype(int)
        color = depth_color[pt[1], pt[0], :].astype(
            int)  # Conert to 0 to 1 range
        image = cv2.circle(image, center=tuple(
            pt), radius=radius, color=color.tolist(), thickness=thickness)

    if image_path != "":
        cv2.imwrite(image_path, image)
        if verbose:
            print('Saved image to', image_path)

    return image


def numpy_to_pcd(array, filename):
    """
    Convert a Nx3 numpy array to a PCD file.

    Parameters:
    array (numpy.ndarray): Nx3 numpy array containing point cloud data.
    filename (str): The output filename for the PCD file.
    """
    import numpy as np
    assert array.shape[1] == 3, "Input array must be Nx3."

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(array)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(array)}
DATA ascii
"""
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, array, fmt='%f %f %f')


def show_bev_map(bev_feats, bev_densities, bev_ids=[0]):
    """
    Shows a BEV map of the given point cloud data next to the colormap of the
    BEV map level features.

    Inputs:
        bev_feats - [B, N, F, H, W] tensor of BEV map level features
        bev_densities - [B, N, 3, H, W] tensor of BEV map densities
    Outputs:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import cv2
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _, _, F, H, W = bev_feats.shape

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    full_rgb_feats = None
    full_rgb_densities = None
    for bev_idx in bev_ids:
        # Average the features and densities over the batch dimension
        fbev_feats = bev_feats[bev_idx].detach()
        fbev_densities = bev_densities[bev_idx].detach()

        fbev_feats = torch.mean(fbev_feats, dim=0)
        fbev_densities = torch.mean(fbev_densities, dim=0)

        # Center and Normalize features before pca reduction
        fbev_feats = fbev_feats.permute(1, 2, 0).reshape(-1, F)
        feat_min = fbev_feats.min(dim=0)[0]
        feat_max = fbev_feats.max(dim=0)[0]
        fbev_feats = (fbev_feats - feat_min) / (feat_max - feat_min)
        reduction_mat = fe.compute_pca_reduction(
            fbev_feats, 100000, 3
        )

        # Normalize features once more to 0 to 1 range
        fbev_feats = fbev_feats @ reduction_mat
        feat_min = fbev_feats.min(dim=0).values
        feat_max = fbev_feats.max(dim=0).values
        rgb_feats = (fbev_feats - feat_min) / (feat_max - feat_min)
        rgb_feats = (255 * rgb_feats)
        rgb_feats = rgb_feats.reshape(H, W, 3)
        rgb_feats = rgb_feats.cpu().numpy()

        fbev_densities = fbev_densities.cpu().numpy()
        fbev_densities = (fbev_densities - fbev_densities.min()) / \
            (fbev_densities.max() - fbev_densities.min())

        full_rgb_feats = full_rgb_feats + \
            rgb_feats if full_rgb_feats is not None else rgb_feats
        full_rgb_densities = full_rgb_densities + \
            fbev_densities if full_rgb_densities is not None else fbev_densities

    # Convert bev density map from [0, MAX_DENSITY] to [0, 255] range
    full_rgb_feats = cv2.normalize(
        full_rgb_feats, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    full_rgb_feats = cv2.applyColorMap(full_rgb_feats, cv2.COLORMAP_TURBO)

    full_rgb_densities = cv2.normalize(
        full_rgb_densities, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    full_rgb_densities = cv2.applyColorMap(
        full_rgb_densities, cv2.COLORMAP_TURBO)

    # Use matplotlib to save the images side by side
    # Assuming input_image is a PIL image or numpy array
    axs[0].imshow(full_rgb_feats)
    axs[0].set_title("PCA Features")
    axs[1].imshow(full_rgb_densities)
    axs[1].set_title("Point Cloud Densities")
    plt.tight_layout(pad=0)
    plt.savefig("bev_feats.png")
    plt.close()
    import pdb
    pdb.set_trace()


def manual_normalization(img, min, max, alpha=0, beta=255):
    """
    Manually normalizes the image from [min, max] to [alpha, beta]
    """
    return alpha + (beta - alpha) * (img - min) / (max - min)


def visualize_bev_label(
    task, features, filepath=None, remap_labels=True, batch_idx=0, kwargs=None
):
    """
    Reads a BEV label and prediction map and displays them side by side. Uses REMAP_SEM_ID_TO_COLOR
    for the mapping between class indices to RGB colors

    Inputs:
        task - str, task name
        features - [2, B, H, W] tensor of BEV features (pred and gt) can be [2, B, F, H, W] if doing FSC_LABEL_DIR
        filepath - path to save the image
    Outputs:
        None
    """
    H, W = features[0][batch_idx].shape[-2:]
    if task == SSC_LABEL_DIR or task == SOC_LABEL_DIR or task == SAM_DYNAMIC_LABEL_DIR:
        bev_pred = features[0][batch_idx].detach().cpu().numpy()
        bev_label = features[1][batch_idx].detach().cpu().numpy()

        # Reduce from [1, H, W] to [H, W] if label was not squeezed
        bev_pred = bev_pred.squeeze()
        bev_label = bev_label.squeeze()

        if task == SSC_LABEL_DIR:
            sem_id_to_color = REMAP_SEM_ID_TO_COLOR if remap_labels else SEM_ID_TO_COLOR
        elif task == SOC_LABEL_DIR:
            sem_id_to_color = REMAP_OBJ_ID_TO_COLOR if remap_labels else REMAP_OBJ_ID_TO_COLOR
        else:
            sem_id_to_color = SAM_DYNAMIC_COLOR_MAP
        
        # Convert to RGB
        id2rgb = np.array(sem_id_to_color, dtype=np.uint8)
        pred_rgb_map = id2rgb[bev_pred]
        gt_rgb_map = id2rgb[bev_label]
    elif task == ELEVATION_LABEL_DIR:
        bev_pred = features[0][batch_idx].detach().cpu().numpy()
        bev_label = features[1][batch_idx].detach().cpu().numpy()

        # Normalizes img from original [min(img1, img2), max(img1, img2)] -> [0, 255]
        global_min = min(bev_pred.min(), bev_label.min())
        global_max = max(bev_pred.max(), bev_label.max())

        pred_rgb_map = manual_normalization(bev_pred, global_min, global_max)
        pred_rgb_map = cv2.applyColorMap(
            pred_rgb_map.astype(np.uint8), cv2.COLORMAP_TURBO)

        gt_rgb_map = manual_normalization(bev_label, global_min, global_max)
        gt_rgb_map = cv2.applyColorMap(
            gt_rgb_map.astype(np.uint8), cv2.COLORMAP_TURBO)
    elif task == SAM_LABEL_DIR:
        bev_pred = make_labels_contiguous_vectorized(features[0][batch_idx])
        bev_label = make_labels_contiguous_vectorized(features[1][batch_idx])
        bev_pred = bev_pred.detach().cpu().numpy()
        bev_label = bev_label.detach().cpu().numpy()

        # Generate cmap
        num_labels = len(np.unique(bev_label))-1
        cmap = generate_cmap(num_labels, use_th=False)[:, :3]
        # cmap = generate_color_map(num_labels)
        cmap = np.concatenate((np.array([[0, 0, 0]]), cmap), axis=0)
        cmap = (cmap * 255).astype(np.uint8)

        # Convert to RGB
        pred_rgb_map = cmap[bev_pred]
        gt_rgb_map = cmap[bev_label]
        # # Convert to RGB
        # id2rgb = np.array(REMAP_OBJ_ID_TO_COLOR, dtype=np.uint8)
        # pred_rgb_map = id2rgb[bev_pred]
        # gt_rgb_map = id2rgb[bev_label]
    elif task == FSC_LABEL_DIR:
        bev_pred = features[0][batch_idx].squeeze()
        bev_label = features[1][batch_idx].squeeze()
        # make_labels_contiguous_vectorized(
        #     features[0][batch_idx].squeeze())
        # bev_label = make_labels_contiguous_vectorized(
        #     features[1][batch_idx].squeeze())

        # Perform PCA reduction on both the prediction and ground truth
        F, H, W = bev_pred.shape
        bev_pred = bev_pred.permute(1, 2, 0).reshape(H*W, F)
        bev_label = bev_label.permute(1, 2, 0).reshape(H*W, F)
        bev_full = torch.cat((bev_pred, bev_label), dim=0).float()

        if kwargs is not None:
            reduction_dict = kwargs["reduction_dict"]
            reduction_mat = reduction_dict["reduction_mat"]
            rgb_min = reduction_dict["rgb_min"]
            rgb_max = reduction_dict["rgb_max"]
            bev_rgb_map = bev_full @ reduction_mat
        else:
            reduction_mat = fe.compute_pca_reduction(bev_full, 100000, 3)
            bev_rgb_map = bev_full @ reduction_mat
            rgb_min = bev_rgb_map.min(dim=0).values
            rgb_max = bev_rgb_map.max(dim=0).values
        bev_rgb_map = (bev_rgb_map - rgb_min) / (rgb_max - rgb_min)
        bev_rgb_map = bev_rgb_map.reshape(2, H, W, 3)

        pred_rgb_map = (bev_rgb_map[0] * 255).detach().cpu().numpy().astype(np.uint8)
        gt_rgb_map = (bev_rgb_map[1] * 255).detach().cpu().numpy().astype(np.uint8)
    elif task == TRAVERSE_LABEL_DIR:
        label_type = kwargs['label_type']
        bev_pred = features[0][batch_idx]
        bev_label = features[1][batch_idx]
        if label_type == SAM_LABEL_DIR:
            bev_label = make_labels_contiguous_vectorized(bev_label)

        bev_pred = bev_pred.detach().squeeze().cpu().numpy()
        bev_label = bev_label.detach().squeeze().cpu().numpy()
        if label_type == SAM_LABEL_DIR:
            assert len(np.unique(bev_label)) - \
                1 == np.max(bev_label), f"SAM labels skipping indices"
            # Generate cmap
            num_labels = len(np.unique(bev_label))-1
            cmap = generate_cmap(num_labels, use_th=False)[:, :3]
            cmap = np.concatenate((np.array([[0, 0, 0]]), cmap), axis=0)
            cmap = (cmap * 255).astype(np.uint8)
            gt_rgb_map = cmap[bev_label.astype(np.uint8)].astype(np.uint8)
        else:
            # SSC Label RGB
            sem_id_to_color = REMAP_SEM_ID_TO_COLOR if remap_labels else SEM_ID_TO_COLOR
            id2rgb = np.array(sem_id_to_color, dtype=np.uint8)
            gt_rgb_map = id2rgb[bev_label.astype(np.uint8)].astype(np.uint8)

        # Reward Map Grayscale but in RGB format
        bev_pred = cv2.normalize(
            bev_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        pred_rgb_map = cv2.cvtColor(
            bev_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if kwargs and 'fov_mask' in kwargs:
            fov_mask = kwargs['fov_mask'].detach().squeeze().cpu().numpy()
            fov_mask = np.stack([fov_mask]*3, axis=-1)
            gt_rgb_map = gt_rgb_map * fov_mask
            pred_rgb_map = pred_rgb_map * fov_mask
    elif task == "embedding":
        id2rgb = np.array(sem_id_to_color, dtype=np.uint8)
        pred_rgb_map = bev_pred.astype(np.uint8)
        gt_rgb_map = id2rgb[bev_label]
    elif task == "textonly":
        # This overrides the default behavior and just draws text on the input images
        # Expects features: [2, H, W, 3] np array
        pred_rgb_map = features[0]
        gt_rgb_map = features[1]
        H, W = pred_rgb_map.shape[-3:-1]
    else:
        raise ValueError(f"Invalid color_map {color_map}")

    # Save image
    pred_text_location = (10, 15)
    gt_text_location = (W + 10, 15)
    img = np.concatenate((pred_rgb_map, gt_rgb_map), axis=1)
    img = cv2.putText(
        img, 'Pred', pred_text_location,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    img = cv2.putText(
        img, 'GT', gt_text_location,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    if filepath is not None:
        cv2.imwrite(filepath, img)
    else:
        return img


def show_elevation_map(elevation_map, color_scale="relative", filepath="elevation_map.png"):
    """
    Displays the elevation map as an RGB image.

    Inputs:
        elevation_map  - [B, H, W] PyTorch tensor or [H, W] NumPy array of elevation values
        color_scale    - either "relative" or "absolute":
                            * "relative": scale data range based on current min/max
                            * "absolute": clip and scale data to a fixed [ABS_MIN, ABS_MAX]
        filepath       - path to save the resulting image

    Outputs:
        norm_elevation - the 8-bit colorized elevation map (H x W x 3)
    """
    # If input is a PyTorch tensor, convert it to NumPy
    if isinstance(elevation_map, torch.Tensor):
        elevation_map = elevation_map[0].detach().cpu().numpy()

    # Remove invalid values (inf) by setting them to 0
    invalid_mask = (elevation_map == np.inf)
    elevation_map[invalid_mask] = 0
    # color_scale = "relative"
    if color_scale == "relative":
        # Use OpenCV's built-in min-max normalization
        norm_elevation = cv2.normalize(elevation_map, None, 
                                       alpha=0, beta=255, 
                                       norm_type=cv2.NORM_MINMAX, 
                                       dtype=cv2.CV_8UC1)
    elif color_scale == "absolute":
        # Predefined absolute min/max for clipping
        ABS_MIN, ABS_MAX = -2.0, 8.0  # Adjust to your desired range
        print("Elevation map min/max:", elevation_map.min(), elevation_map.max())
        clipped = np.clip(elevation_map, ABS_MIN, ABS_MAX)
        # Map [ABS_MIN, ABS_MAX] -> [0, 255]
        scaled = (clipped - ABS_MIN) / (ABS_MAX - ABS_MIN)
        scaled = (scaled * 255).astype(np.uint8)
        norm_elevation = scaled
    else:
        raise ValueError(f"Unknown color_scale option: {color_scale}")

    # Apply the TURBO colormap
    norm_elevation = cv2.applyColorMap(norm_elevation, cv2.COLORMAP_TURBO)

    # Save and return
    cv2.imwrite(filepath, norm_elevation)
    return norm_elevation

def visualize_pc_3d(pc, filepath=None):
    """
    Visualizes the local point cloud in 3D using vispy
    Inputs:
        pc_th - [N, 3+F] tensor of point cloud
    """
    xyz = pc[:, :3]
    xyz = torch.cat((xyz, torch.ones_like(xyz[:, 0]).unsqueeze(-1)), dim=1)

    # Convert from LiDAR to BEV Coordinate System
    lidar2map = torch.tensor([
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).float().to(xyz.device)

    xyz[:, 0] = -xyz[:, 0]  # Reflect over yz plane for visualization
    xyz = (lidar2map @ xyz.T).T
    xyz = xyz.cpu().numpy()[:, :3]

    # Configure visualizer
    from models.blocks.splat_projection import Camera2World
    cam_param = {'scale_factor': 20.9183622824872, 'center': (0.0, 10.0, 0.0), 'fov': 60.0,
                 'elevation': 90, 'azimuth': 0.0, 'roll': 0.0}
    cam2world = Camera2World()
    visualizer_3d = vegt.make_visualizer(256, 256, cam_param=cam_param)
    visualizer_3d.set_points_visible(True)

    visualizer_3d.draw_points(xyz, size=0.1)

    # 3 Save of return image
    pc_vis = visualizer_3d.render()[..., :3]

    pc_vis = cv2.putText(
        np.array(pc_vis), 'Input', (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    if filepath is not None:
        vis_img = Image.fromarray(pc_vis)
        vis_img.save(filepath)
    else:
        return pc_vis


def visualize_rgbd_bev(rgbd_th, p2p_th, map_res, map_sz, map_origin, num_scans=1, num_cams=2, filepath=None):
    """
    Displays a metric accurate BEV map of the given rgbd and pixel2lidar projection matrix.
    Works for multi and single frame inputs. Assumes T=1 when using this function
    Inputs:
        rgbd_th - [B*T*S, 4, H, W] tensor of depth map
        p2p_th - [B*T*S, 4, 3] tensor of projection matrix
        map_res - resolution of the BEV map (cells/m)
        map_sz - size of the BEV map (tuple) (cells)
        map_origin - origin of the BEV map (cells)
        filepath - path to save the image, if None then returns image
    Returns:
        None, if filepath is not None else returns the torch image [3, H, W]
    """
    BTS, C, H, W = rgbd_th.shape
    BT = int(BTS / num_cams)
    S = int(num_cams)
    rgbd_th, p2p_th = rgbd_th.cpu(), p2p_th.cpu()

    assert BTS % num_cams == 0, f'Number of frames should be divisible by number of cameras, got {BTS} frames and {num_cams} cameras'
    assert C == 4, f"Expected 4 channels, got {C}"
    rgbd_th = rgbd_th.reshape(BT, S, C, H, W)  # [BT, S, C, H, W]
    p2p_th = p2p_th.reshape(
        BT, S, p2p_th.shape[-2], p2p_th.shape[-1])  # [BT, S, 4, 3]

    bev_map_rgb = torch.zeros((3, map_sz[0], map_sz[1]), dtype=torch.float32)
    T_lidar_to_bev = torch.tensor([
        [-1, 0, 0, map_origin[0]],
        [0, -1, 0, map_origin[1]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).float()

    from creste.models.blocks.splat_projection import Camera2World
    cam2world = Camera2World()
    for b in range(0, BT):  # Iterate over pairs of frames
        for i in range(num_cams):  # Within each pair, iterate over the two frames
            depth = rgbd_th[b, i, 3, :, :].unsqueeze(
                0).unsqueeze(0) / 1000
            rgb = rgbd_th[b, i, [0, 1, 2], :, :]  # [C, H, W]
            lab = kornia.color.rgb_to_lab(
                rgb.unsqueeze(0)).squeeze()  # [3, H, W]
            p2p = p2p_th[b, i, :, :].unsqueeze(0).unsqueeze(0)  # [4, 3]
            xyz = cam2world((depth, p2p)).squeeze().permute(
                1, 2, 0)  # [H, W, 3]
            xyz = torch.cat((xyz, torch.ones_like(
                xyz[:, :, 0]).unsqueeze(-1)), dim=-1)

            # Convert from LiDAR to BEV Coordinate System\
            xyz[:, :, 0] /= map_res
            xyz[:, :, 1] /= map_res
            xyz = xyz.view(-1, 4)
            rgb = rgb.permute(1, 2, 0).view(-1, 3)
            lab = lab.permute(1, 2, 0).view(-1, 3)
            bev_xyz = torch.matmul(T_lidar_to_bev, xyz.T).T

            depth_mask = (bev_xyz[:, 0] >= 0) & (bev_xyz[:, 0] < map_sz[0]) & (
                bev_xyz[:, 1] >= 0) & (bev_xyz[:, 1] < map_sz[1])

            bev_xyz = bev_xyz[depth_mask]   # N, 4
            bev_rgb = rgb[depth_mask]  # N, 3
            bev_lab = lab[depth_mask]  # N, 3

            # Take the average of the rgb values for each bev cell
            bev_xy = bev_xyz[:, :2].long()
            bev_xy[:, 0] = torch.clamp(bev_xy[:, 0], min=0, max=map_sz[0] - 1)
            bev_xy[:, 1] = torch.clamp(bev_xy[:, 1], min=0, max=map_sz[1] - 1)

            bev_xy_1d = bev_xy[:, 0] * map_sz[1] + bev_xy[:, 1]

            bev_map_rgb_1d = torch_scatter.scatter(
                src=bev_lab.T, index=bev_xy_1d.long(), dim=1, reduce="mean",
                dim_size=map_sz[0] * map_sz[1]
            )

            # Create rgb image from bev_map_rgb
            bev_map_lab = bev_map_rgb_1d.view(3, map_sz[0], map_sz[1])

            unseen_mask = (bev_map_rgb == 0).all(dim=0)
            bev_map_rgb[:, unseen_mask] = kornia.color.lab_to_rgb(
                bev_map_lab.unsqueeze(0)).squeeze()[:, unseen_mask]  # [1, 3, H, W] -> [3, H, W]

    if filepath is not None:
        cv2.imwrite("test.png", (bev_map_rgb.permute(1, 2, 0) *
                    255).numpy().astype(np.uint8))

        # Visualize rgb image to sanity check
        cv2.imwrite("rgb.png", (rgb.view(H, W, 3) *
                    255).cpu().numpy().astype(np.uint8))
    return bev_map_rgb


def visualize_rgbd_3d(rgbd_th, p2p_th, num_scans=1, num_cams=2, filepath=None, do_z_filtering=False, z_max=2):
    """
    Displays a qualitatively accurate BEV map of the given rgbd and 2D to LiDAR projection matrix. Assume T=1 when using this function. Works for multi and single frame inputs

    Inputs:
        rgbd_th - [B*T*S, 4, H, W] tensor of depth map
        p2p_th - [B*T*S, 4, 3] tensor of projection matrix
        filepath - path to save the image, if None then returns image
    Returns:
        None, if filepath is not None else returns the image
    """
    BTS, C, H, W = rgbd_th.shape
    BT = int(BTS / num_cams)
    S = int(num_cams)

    assert BTS % num_cams == 0, f'Number of frames should be divisible by number of cameras, got {BTS} frames and {num_cams} cameras'
    assert C == 4, f"Expected 4 channels, got {C}"
    rgbd_th = rgbd_th.reshape(BT, S, C, H, W)  # [BT, S, C, H, W]
    p2p_th = p2p_th.reshape(
        BT, S, p2p_th.shape[-2], p2p_th.shape[-1])  # [BT, S, 4, 3]

    # Configure visualizer
    from models.blocks.splat_projection import Camera2World
    # cam_param = {'scale_factor': 17.9183622824872, 'center': (0.0, 5.0, 0.0), 'fov': 60.0,
    #                 'elevation': 20, 'azimuth': 0.0, 'roll': 0.0}
    cam_param = {'scale_factor': 9, 'center': (0.0, 4.0, 0.0), 'fov': 60.0,
                 'elevation': 90, 'azimuth': 0.0, 'roll': 0.0}

    cam2world = Camera2World()
    visualizer_3d = vegt.make_visualizer(256, 256, cam_param=cam_param)
    visualizer_3d.set_points_visible(True)

    # Accumulate points and colors from all frames
    all_xyz = []
    all_rgb = []

    # N_f = B // 2
    for b in range(0, BT):  # Iterate over pairs of frames
        for i in range(num_cams):  # Within each pair, iterate over the two frames
            depth = rgbd_th[b, i, 3, :, :].unsqueeze(
                0).unsqueeze(0) / 1000  # Convert to meters
            depth_mask = (depth > 0).squeeze()  # [1, 1, H, W] -> [H, W]
            rgb = rgbd_th[b, i, [2, 1, 0], :, :].permute(1, 2, 0)  # [H, W, C]
            p2p = p2p_th[b, i, :, :].unsqueeze(0).unsqueeze(0)  # [4, 3]

            xyz = cam2world((depth, p2p)).squeeze().permute(1, 2, 0)  # HxWx3
            if do_z_filtering:
                z_mask = xyz[:, :, 2] < z_max
                z_mask = z_mask.unsqueeze(2)
                xyz = xyz*z_mask

            xyz = xyz[depth_mask].reshape(-1, 3)
            rgb = rgb[depth_mask].reshape(-1, 3)

            xyz = torch.cat((xyz, torch.ones_like(
                xyz[:, 0]).unsqueeze(-1)), dim=1)

            # Convert from LiDAR to BEV Coordinate System
            lidar2map = torch.tensor([
                [0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]).float().to(xyz.device)
            # lidar2map = torch.tensor([
            #     [0, 1, 0, 0],
            #     [0, 0, -1, 0],
            #     [-1, 0, 0, 0],
            #     [0, 0, 0, 1]
            # ]).float().to(xyz.device)

            xyz[:, 0] = -xyz[:, 0]  # Reflect over yz plane for visualization
            xyz = (lidar2map @ xyz.T).T
            xyz = xyz.cpu().numpy()[:, :3]
            # numpy_to_pcd(xyz[:, :3], f'aggregated{i}.pcd')

            all_xyz.append(xyz)
            all_rgb.append(rgb.cpu().numpy())
            # all_xyz.insert(0, xyz)
            # all_rgb.insert(0, rgb.cpu().numpy())

    # Aggregate all points and colors
    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)

    visualizer_3d.draw_points(all_xyz, size=3, colors=all_rgb)

    # 3 Save of return image
    pc_vis = visualizer_3d.render()[..., :3]

    pc_vis = cv2.putText(
        np.array(pc_vis), 'Input', (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    if filepath is not None:
        vis_img = Image.fromarray(pc_vis)
        vis_img.save(filepath)
    else:
        return pc_vis


# def visualize_unified_3d_scene(inputs, vis_dict, batch_idx=0, save_path=None):
#     """
#     Unified 3d visualization viewer. Given dictionary of inputs and visualization
#     args, visualizes the 3d scene mesh with vispy

#     Inputs:
#         inputs:
#             - 'semantic' - [B, 64, H, W] semantic predictions
#             - 'elevation' - [B, 1, H, W] elevation predictions
#         vis_args: dictionary containing how to visualize each input key
#             {
#                 key: {
#                     'type': 'elevation',
#                     'color_map': 'turbo',
#                     'save_path': 'elevation.png'
#                 }
#             }
#     """
#     for key, val in vis_dict.items():
#         if val['type'] == 'elevation':

#             elevation = inputs['elevation'][batch_idx].detach().cpu().numpy()
#             elevation = elevation.squeeze()
#             elevation = cv2.normalize(
#                 elevation, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#             elevation = cv2.applyColorMap(elevation, cv2.COLORMAP_TURBO)

#             # Save image
#             pred_text_location = (10, 15)
#             img = cv2.putText(
#                 elevation, 'Elevation', pred_text_location,
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
#             )

#             if val['save_path'] is not None:
#                 cv2.imwrite(val['save_path'], img)
#             else:
#                 return img


def visualize_elevation_3d_wrapper(elevation_preds, elevation_gts, batch_idx=0, color_map=None, save_path=None, unoccluded_mask=None):
    """
    Wrapper function to visualize elevation predictions

    Inputs:
        elevation_pred - [B, H, W] predicted elevation
        elevation_gt   - [B, H, W] ground truth elevation
    Ouputs: None
    """
    assert elevation_preds.shape == elevation_gts.shape and elevation_preds.ndim == 3, \
        f'elevation_preds.shape {elevation_preds.shape} != elevation_gts.shape {elevation_gts.shape}'
    elevation_pred = elevation_preds[batch_idx].detach().cpu().numpy()
    elevation_gt = elevation_gts[batch_idx].detach().cpu().numpy()
    H, W = elevation_pred.shape

    # Zero infinite values
    valid_elevation_pred = elevation_pred.copy()
    valid_elevation_gt = elevation_gt.copy()
    valid_elevation_pred[np.isinf(valid_elevation_pred)] = -0.8
    valid_elevation_pred[np.isnan(valid_elevation_pred)] = -0.8
    valid_elevation_gt[np.isinf(valid_elevation_gt)] = -0.8
    valid_elevation_gt[np.isnan(valid_elevation_gt)] = -0.8
    # assert np.isinf(valid_elevation_pred).sum() == 0, f"Found infinite values in elevation_pred, only nan values allowed"

    if color_map is None:
        global_min = min(valid_elevation_pred.min(), valid_elevation_gt.min())
        global_max = max(valid_elevation_pred.max(), valid_elevation_gt.max())

        # 2 Generate colormaps
        pred_rgb_map = manual_normalization(
            valid_elevation_pred, global_min, global_max)
        pred_rgb_map = cv2.applyColorMap(
            pred_rgb_map.astype(np.uint8), cv2.COLORMAP_TURBO)
        gt_rgb_map = manual_normalization(
            valid_elevation_gt, global_min, global_max)
        gt_rgb_map = cv2.applyColorMap(
            gt_rgb_map.astype(np.uint8), cv2.COLORMAP_TURBO)
    if unoccluded_mask is not None:
        # Blend colormmap with unoccluded mask
        unoccluded_mask = unoccluded_mask[batch_idx].squeeze(
        ).unsqueeze(-1).cpu().numpy()
        pred_rgb_map = np.where(unoccluded_mask, pred_rgb_map, cv2.addWeighted(
            pred_rgb_map, 0.7, np.zeros_like(pred_rgb_map), 0.3, 0))
        gt_rgb_map = np.where(unoccluded_mask, gt_rgb_map, cv2.addWeighted(
            gt_rgb_map, 0.7, np.zeros_like(gt_rgb_map), 0.3, 0))

    # 3 Save predicted elevation map
    img_pred = vegt.visualize_label(
        (elevation_pred, pred_rgb_map / 255.0, None, H, W))
    img_gt = vegt.visualize_label(
        (elevation_gt, gt_rgb_map / 255.0, None, H, W))

    # Combine the two images and add prediction/GT text
    combined_img = np.concatenate(
        (np.array(img_pred), np.array(img_gt)), axis=1)
    # Truncate the half because it is all black
    combined_img = combined_img[H*1//3:, :, :]
    combined_img = cv2.putText(
        combined_img, 'Pred', (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    combined_img = cv2.putText(
        combined_img, 'GT', (W + 10, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    if save_path is not None:
        cv2.imwrite(save_path, combined_img)
    else:
        return combined_img


def draw_text_on_image(image, text, location=(10, 15), font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=0.5, color=(255, 255, 255), thickness=1):
    """
    Draws text on an image at a specified location.

    Parameters:
    - image: HxWx3 NumPy array representing an RGB image.
    - text: The text to draw on the image.
    - location: A tuple of two elements (x, y) representing the location to draw the text.
    - font: The font to use for the text.
    - font_scale: The scale of the font.
    - color: A tuple of three elements (R, G, B) representing the color of the text.
    - thickness: The thickness of the text.

    Returns:
    - result_image: HxWx3 NumPy array of the image with the text drawn.
    """
    result_image = cv2.putText(image, text, location, font, font_scale,
                               color, thickness, cv2.LINE_AA)

    return result_image


def generate_color_map(N):
    # HSV values to step through hues evenly and convert to RGB
    rgb_colors = []

    hue_partition = 1.0 / (N + 1)
    for val in range(0, N):
        (r, g, b) = colorsys.hsv_to_rgb(hue_partition * val, 1.0, 1.0)
        rgb_colors.append([r, g, b])

    return np.array(rgb_colors)


def apply_alpha_to_image(image, alpha_mask, background):
    """
    Applies an alpha mask to an RGB image, blending it with a specified background color.

    Parameters:
    - image: HxWx3 NumPy array representing an RGB image.
    - alpha_mask: HxW NumPy array where each value is between 0 and 1, representing the alpha value for that pixel.
    - background_color: A tuple of three elements (R, G, B) representing the background color.

    Returns:
    - result_image: HxWx3 NumPy array of the image after applying the alpha mask.
    """
    # Ensure alpha_mask is expanded to shape HxWx3 for broadcasting with the image
    alpha_expanded = np.expand_dims(alpha_mask, axis=-1)

    # Apply the alpha blending formula
    result_image = alpha_expanded * image + (1 - alpha_expanded) * background

    return result_image


def draw_bev_heatmap(heatmap, img, color_map):
    """Draw 2D BEV bounding box on an image.

    Args:
        heatmap: np.array, HxWxC array of the heatmap.
        img (numpy array): (HxWx3) Image to draw the bounding box on.
        color (tuple, optional): Color of the bounding box. Defaults to (0, 255, 0).

    Returns:
        numpy array: Image with bounding box drawn.
    """
    # Draw the bounding box
    heatmap_label = np.argmax(heatmap, axis=-1)
    heatmap_color = color_map[heatmap_label]
    heatmap_alpha = np.max(heatmap, axis=-1).astype(np.float32)

    img = apply_alpha_to_image(heatmap_color, heatmap_alpha, img)

    return img


def draw_bev_bbox(img, bbox, color, center=None, thickness=1):
    """Draw 2D BEV bounding box on an image.

    Args:
        corners: np.array, Bx4x2 array of the corners of the BEV bounding box.
        img (numpy array): (HxWx3) Image to draw the bounding box on.
        color (tuple, optional): Color of the bounding box. Defaults to (0, 255, 0).

    Returns:
        numpy array: Image with bounding box drawn.
    """
    # Add fifth point to enclose bbox
    bbox = np.concatenate((bbox, np.expand_dims(bbox[:, 0, :], 1)), axis=1)

    # Draw the bounding box
    for b in range(bbox.shape[0]):
        for i in range(bbox.shape[1]-1):
            img = cv2.line(img, tuple(bbox[b, i]), tuple(
                bbox[b, i+1]), color[b].tolist(), thickness, lineType=cv2.LINE_AA)
        if center is not None:
            img = cv2.circle(img, tuple(center[b, 0].astype(
                int)), radius=2, color=color[b].tolist(), thickness=thickness)

    return img


def visualize_bev_poses(batch_poses, img=None, batch_idx=0, color=(255, 0, 0), thickness=1, indexing='ij'):
    """
    Inputs:
        poses - [B, T, 3, 3] tensor of 3x3 of poses in a BEV map
            can also be [B, T, 2] tensor of poses in pixels
            T is the number of poses, B is the batch size, we visualize only the first item in batch
        img - [H, W, 3] tensor of BEV RGB map
    Outputs:
        img - [H, W, 3] tensor of BEV RGB map with poses drawn
    """
    # Draw the poses using matplotlib
    poses = batch_poses[batch_idx]
    if type(batch_poses) == torch.Tensor:
        poses = poses.detach().cpu().numpy()
    if img is None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Draw poses on image (assume poses are in pixels)
    for i in range(0, poses.shape[0]-1):
        p1 = poses[i]
        p2 = poses[i+1]
        if batch_poses.ndim < 4:
            x1, y1 = p1
            x2, y2 = p2
            if indexing=='xy':
                x1, y1 = y1, x1
                x2, y2 = y2, x2
        else:
            x1, y1 = p1[1, 2], p1[0, 2]
            x2, y2 = p2[1, 2], p2[0, 2]
        img = cv2.circle(img, (int(x1), int(y1)), radius=1, color=color, thickness=thickness)
        # img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=thickness)

    if img is None:
        cv2.imwrite('poses.png', img)
        # print("Distance between first and last pose ", np.linalg.norm(poses[0, :2, 2] - poses[-1, :2, 2]))
    return img


def visualize_bev_policy(policy, start=None, goal=None, img=None, batch_idx=0):
    """
    Visualizes a policy on an 8-connected grid using colors for each action.

    Args:
        policy: Tensor of shape (B, A, H, W), where A = 8 corresponds to movement
                direction probabilities in an 8-connected grid.
    """
    B, A, H, W = policy.shape

    # Define BGR colors for each action (0-7) for OpenCV
    action_colors_bgr = {
        0: (128, 0, 128),  # Purple
        1: (255, 0, 0),    # Blue
        2: (255, 255, 0),  # Cyan
        3: (0, 255, 0),    # Green
        4: (0, 255, 255),  # Yellow
        5: (0, 128, 255),  # Orange
        6: (0, 0, 255),    # Red
        7: (203, 192, 255)  # Pink
    }

    index_to_action = [
        "UP_LEFT", "UP", "UP_RIGHT", "LEFT", "RIGHT", "DOWN_LEFT", "DOWN", "DOWN_RIGHT"
    ]

    # Get the most likely action at each (H, W) location
    most_likely_action = torch.argmax(policy[batch_idx], dim=0).cpu().numpy()

    # Create a blank image with 3 channels (for BGR colors)
    bev_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Populate the image with colors based on actions
    for action, color in action_colors_bgr.items():
        bev_image[most_likely_action == action] = color

    # Optionally, draw the start and goal points using OpenCV
    if goal is not None:
        goal = goal[batch_idx].detach().cpu().numpy().astype(int)
        cv2.circle(bev_image, (goal[1], goal[0]), radius=3, color=(
            0, 0, 255), thickness=-1)  # Red circle for the goal

    if start is not None:
        start = start[batch_idx].detach().cpu().numpy().astype(int)
        cv2.circle(bev_image, (start[1], start[0]), radius=3, color=(
            0, 255, 0), thickness=-1)  # Green circle for the start

    # Convert the OpenCV image to RGB for plotting with Matplotlib
    bev_image_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)

    # Create a Matplotlib figure to add the legend
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Policy Visualization for Batch {batch_idx}')

    # Show the OpenCV image in the Matplotlib figure
    ax.imshow(bev_image_rgb)

    # Create a legend for action colors
    legend_patches = [mpatches.Patch(color=np.array(
        color)/255.0, label=f'{index_to_action[i]}') for i, color in action_colors_bgr.items()]
    ax.legend(handles=legend_patches, loc='upper right')

    # Show the start and goal points if they exist
    if goal is not None:
        ax.plot(goal[1], goal[0], 'ro', label='Goal')

    if start is not None:
        ax.plot(start[1], start[0], 'go', label='Start')

    ax.invert_yaxis()  # Optional: match image convention (top-left is (0, 0))
    plt.grid(True)

    # Convert the plot to an OpenCV image
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    rgba = buf.reshape(h, w, 4)          # H × W × 4
    img  = rgba[..., :3]                 # drop alpha if not needed
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)  # Close the figure to free memory

    return img_bgr


def visualize_action_image(img, actions_in, transform, batch_idx=0):
    """
    Projects the input actions to the original rgb image and visualizes them
    Inputs:
        img - [B, 3, H, W] tensor of RGB image
        actions_in - [B, T, 3] tensor of input actions, T is time horizon
        transform - [B, 3, 4] xyz to uv projection matrix
    Outputs:
        img - [B, 3, H, W] tensor of RGB image with actions drawn
    """
    pass


def visualize_action_label(actions_in, pred, gt, transform):
    """
    Inputs:
        img - [B, H, W, 3] tensor of BEV RGB map
        actions_in - [B, H, 2] tensor of input actions, H is past horizon
        pred - [B, F, 2] tensor of predicted actions, F is future horizon
        gt - [B, F, 2]
        transform - [B, 3, 3] tensor of 3x3 transformation matrix
    """
    actions_in = actions_in[0].detach().cpu().numpy()
    pred = pred[0].detach().cpu().numpy()
    gt = gt[0].detach().cpu().numpy()

    # Convert inputs to homogeneous coordinates
    actions_in = np.concatenate(
        (actions_in, np.ones((actions_in.shape[0], 1))), axis=1)
    pred = np.concatenate((pred, np.ones((pred.shape[0], 1))), axis=1)
    gt = np.concatenate((gt, np.ones((gt.shape[0], 1))), axis=1)

    # Convert from LiDAR to BEV Coordinate System
    actions_in = (transform @ actions_in.T).T
    pred = (transform @ pred.T).T
    gt = (transform @ gt.T).T

    # Conert back to xy coordinates
    actions_in = actions_in[:, :2]
    pred = pred[:, :2]
    gt = gt[:, :2]

    # Draw the input actions using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create fixed axes ranges
    ax.set_xlim(12.5, 13)
    ax.set_ylim(12.5, 13)

    # plot 2d actions on matplotlib
    ax.plot(actions_in[:, 0], actions_in[:, 1], 'ro', label='Input')
    ax.plot(pred[:, 0], pred[:, 1], 'bo', label='Pred')
    ax.plot(gt[:, 0], gt[:, 1], 'go', label='GT')
    ax.legend()

    # Save the image
    plt.savefig('actions.png')
    plt.close()
    fig.clear()
    img = cv2.imread('actions.png')

    return img


def visualize_dino_feature(rgbd_img_th, features_th, img_path="test.png"):
    """
    Inputs:
        rgb_img - [B, 3/4, H, W] tensor of RGB/D image, expects image to be from 0 to 1 and BGR format
        features - [B, F, H, W] tensor of feature image
    """
    # Normalizes the feature across the batch dimension
    B, C, H, W = features_th.shape
    assert features_th.ndim == 4, f"Expected 4D tensor, got {features_th.ndim}"

    # Batch pca reduction
    features = features_th.cuda()
    # Reduce features to 3D
    features = features.permute(0, 2, 3, 1).flatten(0, 2)  # [B*H*W, F]

    # Convert the features to image space
    reduction_mat, feat_color_min, feat_color_max = fe.get_robust_pca(
        features, features.shape[0], 3
    )

    full_image = None
    for b in range(B):
        rgb_img = (rgbd_img_th[b].permute(1, 2, 0)[
            :, :, :3].cpu().numpy()*255.0).astype(np.uint8)
        features = features_th[b].cuda().permute(
            1, 2, 0).flatten(0, 1)  # [F, H, W] -> [H*W, F]
        lowrank_feature = features @ reduction_mat
        lowrank_feature = lowrank_feature.reshape(
            H, W, -1
        ).squeeze()
        norm_lowrank_feature = (
            (lowrank_feature - feat_color_min) / (feat_color_max - feat_color_min)).clamp(0, 1)
        rgb_feature = (norm_lowrank_feature.cpu().numpy()
                       * 255).astype(np.uint8)

        feature_img = Image.fromarray(rgb_feature)
        rgb_img = Image.fromarray(rgb_img).resize((W, H), Image.NEAREST)

        alpha = 0.3
        overlay = Image.blend(feature_img, rgb_img, alpha=alpha)

        if full_image is None:
            full_image = overlay
        else:
            full_image = np.concatenate((full_image, overlay), axis=1)
    cv2.imwrite(img_path, full_image)

def write_text_on_image(image_np, text, position=(10, 10), color=(255, 255, 255), fontsize=0.3, thickness=1):
    """
    Writes text on a torch image tensor (HxWx3) at the specified position.

    Args:
        image_np (np.array): Image tensor of shape (HxWx3), representing an RGB image.
        text (str): The text to write on the image.
        position (tuple): (x, y) coordinates for the position of the text.
        color (tuple): The RGB color for the text, defaults to white (255, 255, 255).
        fontsize (int): The font size for the text.
        thickness (int): The thickness of the text.

    Returns:
        np.array: np array with text written on it.
    """
    # Ensure the image is in the correct format (uint8)
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Use OpenCV to write text on the image
    cv2.putText(image_np, text, position, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness, cv2.LINE_AA)

    return image_np

"""BEGIN SAM RELATED VISUALIZATION FUNCTIONS"""


def generate_cmap(size, use_th=False):
    if use_th:
        cmap = torch.rand(size*3).reshape(size, 3).float()
        cmap = torch.cat([cmap, torch.ones((size, 1))], axis=1)
    else:
        cmap = np.random.random(size*3).reshape(size, 3).astype(np.float32)
        cmap = np.concatenate([cmap, np.ones((size, 1))], axis=1)
    return cmap


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()
    return color


def show_masks_on_image(img, labels, img_path=None, cmap=None, alpha=0.5):
    """
    Plots mask on cv2 image

    Inputs:
        img - [B, 3, H, W] torch tensor for image
        mask[B, H, W] torch tensor for label
    """
    img = img[0].cpu().numpy().transpose(1, 2, 0)
    labels = labels[0].cpu().numpy()

    # num_labels = len(np.unique(labels))
    num_labels = np.max(labels)+1
    if cmap is None:
        # [num_labels, 4] with last being alpha
        cmap = generate_cmap(num_labels)

    # labels_mask = F.one_hot(labels, num_labels).permute(1, 2, 0).float()
    labels_rgb = cmap[labels][:, :, :3]
    labels_rgb = (labels_rgb * 255).astype(np.uint8)

    # Blend the label rgb with the image
    img = (img * 255).astype(np.uint8)
    blended_img = cv2.addWeighted(img, 1-alpha, labels_rgb, alpha, 0)
    if img_path is not None:
        cv2.imwrite(img_path, blended_img)
    else:
        return blended_img


"""END SAM RELATED VISUALIZATION FUNCTIONS"""
