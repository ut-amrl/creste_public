import os
import argparse

import cv2
import numpy as np
from PIL import Image

from creste.utils import pointcloud_vis 

def parse_args():
    parser = argparse.ArgumentParser(description='Build semantic map from point clouds')
    parser.add_argument('--cfg', type=str, default='./configs/dataset/coda.yaml')
    parser.add_argument('--out_dir', type=str, default='./postprocess/build_map_outputs/codaelevation')
    args = parser.parse_args()

    return args

def make_visualizer(width, height, cam_param=None):
    if cam_param is None:
        # cam_param = {'scale_factor': 130.9183622824872, 'center': (0.0, 20.0, 0.0), 'fov': 60.0,
        #             'elevation': 20.5, 'azimuth': 0.0, 'roll': 0.0}
        cam_param = {'scale_factor': 125.9183622824872, 'center': (0.0, 80.0, 0.0), 'fov': 60.0,
                    'elevation': 10.5, 'azimuth': 0.0, 'roll': 0.0}
        print("Using default cam params")
    vis = pointcloud_vis.LaserScanVis(width=width, height=height, interactive=False)
    vis.set_camera(cam_param)
    return vis

def visualize_elevation_3d(elevation, colors, width, height, resolution=0.5, cam_param=None, points=None):
    import vispy
    vispy.use('egl')

    if 'visualizer_3d' not in globals():
        global visualizer_3d
        visualizer_3d = make_visualizer(width, height, cam_param)
    
    visualizer_3d.set_mesh_visible(True)
    if np.all(np.isnan(elevation)):
        elevation = elevation.copy()
        elevation[...] = 0

    ref_size = 256
    scale = ref_size / elevation.shape[0]

    visualizer_3d.draw_mesh_grid(elevation * scale*4, ~np.isnan(elevation), scale * resolution, colors)

    if points is not None:
        visualizer_3d.set_points_visible(True)
        visualizer_3d.draw_points(points, size=0.5, colors=(1.0,0,0))
    else:
        visualizer_3d.set_points_visible(False)
    return visualizer_3d.render()[..., :3]
    
def visualize_label(inputs):
    """
    Accepts either string filepath or direct numpy array for label and color

    label - str or np.array - path to elevation label or numpy array (HxW)
    color - str or np.array - path to color image or numpy array (HxWx3)

    Returns:
        vis_img - PIL.Image - visualization of the label and color
    """
    label, color, out_path, height, width = inputs
    if type(label) is str:
        assert os.path.exists(label), f'Label path {label} does not exist'
        assert os.path.exists(color), f'Color path {color} does not exist'

        elevation = np.fromfile(label, dtype=np.float64).reshape((height, width))
        elevation_color = cv2.imread(color) / 255.0 # Convert to 0-1
    else:
        elevation = label
        elevation_color = color
    elevation[elevation == np.inf] = np.nan

    #Flip elevation maps to match coordinate system of visualizer
    elevation = np.flip(elevation, axis=0)
    elevation_color = np.flip(elevation_color, axis=0)

    elev_vis = visualize_elevation_3d(elevation, elevation_color, width, height)
    
    vis_img = Image.fromarray(elev_vis)

    if out_path is not None:
        print(f'Saved visualization to {out_path}')
        vis_img.save(out_path)
    else:
        return vis_img
