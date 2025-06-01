"""
Borrowed from Amirreza Shaban for filtering map eleveation TerrainNet

Maintainer: Arthur K. Zhang
"""
import time
from numba.typed import List
import vispy
# vispy.use('egl')
import cv2
from creste.utils.pointcloud_vis import LaserScanVis
import numpy as np

import torch
import torch_scatter


@torch.no_grad()
def _scatter_min(arr, inds, ntop=1, dim_size=None):
    arr = arr.detach().clone()

    if dim_size is None:
        dim_size = inds.max() + 1

    invalid_ind = len(arr)
    top_min = []
    for n in range(ntop):
        v, vi = torch_scatter.scatter_min(arr, inds, dim_size=dim_size)
        v[vi == invalid_ind] = float("Inf")
        top_min.append(v)

        if n != ntop - 1:
            # remove min
            arr[vi[vi != invalid_ind]] = float("Inf")

    return torch.stack(top_min, axis=1)


def crop_center(arr, h, w):
    sy = (arr.shape[0] - h)//2
    sx = (arr.shape[1] - w)//2
    return arr[..., sy:sy+h, sx:sx+w]


class Map2D(object):
    def __init__(self, width, height, resx, resy):
        '''
            Center of the map is assumed to be at (0,0).
            One can adjust width and height and the map
            resolution.
        '''
        self.width = width  # map width in global coordinate
        self.height = height  # map height in global coordinate
        self.resx = resx
        self.resy = resy

        assert type(self.resx) == int and type(
            self.resy) == int, 'resx and resy should be integers'

    def init_map(self, kw, kh, stride=1):
        '''
            kw, kh: size of the kernel in resolution
        '''

        # last_indx = self.resx - kw
        # last_indy = self.resy - kh
        last_indx = self.resx - 1
        last_indy = self.resy - 1

        # check if the center stays at (0,0)
        if last_indx % stride != 0 or last_indy % stride != 0:
            raise ValueError(
                'Choose the kernel size and stride so the center stays at (0,0)')

        # output map resolution
        resx = int(last_indx / stride + 1)
        resy = int(last_indy / stride + 1)

        # output map cell size
        cellw = stride*self.width/float(self.resx)
        cellh = stride*self.height/float(self.resy)

        # output map
        width = cellw * resx
        height = cellh * resy

        return Map2D(width, height, resx, resy)

    def apply_kernel(self, w, h, stride=1, op='mean'):
        assert (op in ['mean', 'max', 'min', 'var'])

        outmap = self.init_map(w, h, stride)
        # Pad map with zeros
        pad = (w//2, h//2, w//2, h//2)
        unfold = torch.nn.Unfold((h, w), stride=stride, padding=stride)

        dt = unfold(self.map[None])[0]
        dt = dt.view(2, w*h, outmap.resy, outmap.resx)

        # count the number of valid points within each bin
        valid_count = dt[1].sum(axis=0)

        map_mask = valid_count > 0  # 0 is invalid, 1 is valid
        if op == 'mean':
            map_val = (dt[0] * dt[1]).sum(axis=0)
            map_val = map_val / (valid_count+1e-6)
        elif op == 'max':
            map_val = dt[0].detach().clone()
            map_val[dt[1] != 1] = float('-inf')
            map_val = map_val.max(axis=0)[0]
        elif op == 'min':
            map_val = dt[0].detach().clone()
            map_val[dt[1] != 1] = float('inf')
            map_val = map_val.min(axis=0)[0]
        elif op == 'var':
            map_mean = (dt[0] * dt[1]).sum(axis=0) / (valid_count+1e-6)

            # Compute variance E[ (X - E[X])^2 ]
            squared_diff = (dt[0] - map_mean.unsqueeze(0))**2
            map_var = (squared_diff * dt[1]).sum(axis=0) / (valid_count + 1e-6)

            # Return both the variance and mask
            map_val = map_var

        map_val[~map_mask] = 0.0
        outmap.map = torch.stack((map_val, map_mask.float()), axis=0)
        # cv2.imwrite('test.png', cv2.normalize(
        #     map_val.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        return outmap

    def fill(self, points, valid=None, proj_ind=None,
             min_points_per_cell=None, ntop_min=None):
        ''' Fill the map with the points in global coordinate system.
            x,y will be used to to find map cell. The map cell will
            be set by the z value.
            Args:
                points: xyz
        '''
        # First channel holds z value, second channel is mask
        self.map = points.new_zeros((2, self.resy, self.resx))
        self.counts = torch.zeros_like(self.map[0], dtype=torch.int32)
        map_view = self.map[0].view(-1)

        if proj_ind is None:
            proj_ind, valid = self.locs(points, valid)

        if ntop_min:
            min_points_per_cell = min_points_per_cell or 0
            min_points_per_cell = max(min_points_per_cell, ntop_min)

        counts = torch_scatter.scatter_add(torch.ones_like(proj_ind[valid]),
                                           proj_ind[valid],
                                           dim_size=len(map_view))
        self.counts.view(-1)[...] = counts[...]
        if min_points_per_cell:
            good_count = counts >= min_points_per_cell
            # Need to clip proj_ind because some of the elements can be out of range.
            # While clipping produces wrong "good count", but since they are out of range,
            # they are invalid anyway. The end result is still correct.
            valid = valid & good_count[torch.clip(
                proj_ind, 0, self.resx * self.resy - 1)]
            self.counts[~good_count.reshape(self.counts.shape)] = 0

        pt = proj_ind[valid]

        if ntop_min:  # robust min
            top_min = _scatter_min(points[valid, 2],
                                   pt,
                                   ntop=ntop_min,
                                   dim_size=len(map_view))
            median_min = top_min.nanmedian(axis=1)[0]
            # since we have set min_points_per_cell we shoud have enough points
            assert (~torch.any(median_min[pt].isnan()))

            median_min[median_min.isinf()] = 0.0
            map_view[...] = median_min
        else:
            map_view[...] = torch_scatter.scatter_min(
                points[valid, 2],
                pt,
                dim_size=len(map_view))[0]

        self.map[1].view(-1)[pt] = 1.0

    def locs(self, points, inrange=None):
        projx = ((points[:, 0] / self.width + 0.5) * self.resx).to(torch.int64)
        projy = ((points[:, 1] / self.height + 0.5)
                 * self.resy).to(torch.int64)
        proj_ind = projx + projy * self.resx
        inrange_ = ((0 <= projx) & (projx < self.resx) &
                    (0 <= projy) & (projy < self.resy))
        inrange = inrange_ if inrange is None else (inrange & inrange_)
        return proj_ind, inrange

    def query(self, points):
        ind, inrange = self.locs(points)
        inrange_ind = ind[inrange]

        # Read value
        values = torch.zeros_like(inrange, dtype=self.map.dtype)
        values[inrange] = self.map[0].view(-1)[inrange_ind]

        # Read mask
        mask = torch.zeros_like(inrange, dtype=self.map.dtype)
        mask[inrange] = self.map[1].view(-1)[inrange_ind]

        return values, mask.to(torch.bool), inrange


class BinningPostprocess(object):
    def __init__(self, config, device):
        '''
            config: a dictionary with the following structure:
               map:
                    width: 40 # map width in meters
                    height: 40 # map height in meters
                    resx: 100 # map x axis resolution
                    resy: 100 # map y axis resolution
                    nlowest_points: 10 # elevation will be the median
                                         of the 10 lowest points
                    pre_kernel_min_points_per_cell: 2
                    post_kernel_min_points_per_cell: 2
               meanz_kernel:
                    resw = 10
                    resh = 10
                    stride = 5
               threshold:
                    sky = 2.5
        '''

        # map
        width = config['map']['width']
        height = config['map']['height']
        resx = config['map']['resx']
        resy = config['map']['resy']

        # We apply a mean kernel with these parameters to estimate ground z
        self.kernel_resw = config['meanz_kernel']['resw']
        self.kernel_resh = config['meanz_kernel']['resh']
        self.kernel_stride = config['meanz_kernel']['stride']

        if 'threshold' in config:
            self.sky_threshold = config['threshold']['sky']

        self.nlowest_points = config['map'].get('nlowest_points', None)
        self.pre_points_per_cell = config['map'].get(
            'pre_kernel_min_points_per_cell', None)
        self.post_points_per_cell = config['map'].get(
            'post_kernel_min_points_per_cell', None)

        self.device = device

        # build ground map
        self.ground_map = Map2D(width, height, resx, resy)

    def build_map(self, points, op='mean'):
        # build ground points map and estimate ground z
        self.ground_map.fill(points,
                             min_points_per_cell=self.pre_points_per_cell,
                             ntop_min=self.nlowest_points)

        self.minz_ground_map = self.ground_map.apply_kernel(
            self.kernel_resw, self.kernel_resh, self.kernel_stride, op=op)

        if self.post_points_per_cell:
            counts = self.ground_map.counts
            low_count = counts < self.post_points_per_cell
            shape = self.minz_ground_map.map.shape[1:]
            self.minz_ground_map.map[:, crop_center(low_count, *shape)] = 0

    def elevation(self, points):
        # get ground z value for all points and compute their elavation from ground
        # we also update inrange as meanz_ground_map might be smaller
        groundz, valid_groundz, inrange = self.minz_ground_map.query(points)
        elevations = points[:, 2] - groundz

        return elevations, valid_groundz, inrange

    def process_pc(self, points, kernel='mean', ntop_min=None):
        ''' We start off by estimating ground elavation by computing minimum z values for each cell.
            Then, we compute each points distance to the estimated
            ground elavation value and correct the predictions as follows:
                + (LOGIC 0) Any point outside the 2d map boundary is classified as unknown.
                + (LOGIC 1) Any point above the sky_threshold is classified as sky.
            The output is a 1d tensor with shape (n,).
        '''
        self.build_map(points, kernel, ntop_min)

        elevations, valid_groundz, inrange = self.elevation(points)

        # Copy the original preds matrix and update it
        preds = torch.zeros(
            points.shape[0], dtype=torch.int64, device=points.device)

        # LOGIC 1:
        preds[elevations > self.sky_threshold] = 1

        # LOGIC 0: Anything outside the minz traversable map is labeled as unknown
        preds[~inrange | ~valid_groundz] = 2

        return preds


def make_grid_points(width, height, points, labels, bin_group, unique_bin_map, cx, cy,
                     base_z, bin_size, ignore_labels):
    elevation_grid = np.zeros((height, width, bin_size), np.float32)
    mask = np.ones((height, width), np.bool)

    def should_ignore(bin_labels):
        if not ignore_labels:
            return False
        for l in ignore_labels:
            if np.count_nonzero(bin_labels == l) > 0:
                return True
        return False

    for i in range(0, width):
        for j in range(0, height):
            bin_x = cx - width // 2 + i
            bin_y = cy - height // 2 + j
            if (bin_x, bin_y) in unique_bin_map:
                # This bin is non-empty
                point_idxs = bin_group[unique_bin_map[(bin_x, bin_y)]]
                bin_points = points[point_idxs]
                if labels is not None:
                    bin_labels = labels[point_idxs]
                    if should_ignore(bin_labels):
                        mask[j, i] = False
                    else:
                        elevation_grid[j, i] = bin_points[:, 2] - base_z
                else:
                    elevation_grid[j, i] = bin_points[:, 2] - base_z
            else:
                mask[j, i] = False
    return elevation_grid, mask


class Worker(object):
    def __init__(self, dataset_dir, dataset_file, lidar_dirs, map_width, map_height):
        vis = LaserScanVis(width=1280, height=960, interactive=True)
        camera_param = {'scale_factor': 163.05396672619742, 'center': [0.0, 0.0, 0.0], 'fov': 45.0,
                        'elevation': 37.5, 'azimuth': 109.5, 'roll': 0.0}
        vis.set_camera(camera_param)
        vis.set_arrow_visible(True)

        loader = DatasetLoader(dataset_dir, lidar_dirs)
        dataset = dict(np.load(dataset_file, allow_pickle=True, mmap_mode='r'))

        unique_bins = dataset['unique_bins']
        unique_bin_map = dict()
        for i in range(len(unique_bins)):
            x, y = unique_bins[i]
            unique_bin_map[(int(x), int(y))] = i

        self.width = map_width
        self.height = map_height
        self.vis = vis
        self.loader = loader
        self.dataset = dataset
        self.unique_bin_map = unique_bin_map

    def make_vis_img(self, pose_idx, mode, ignore_labels=None):
        dataset = self.dataset
        points = dataset['points']
        labels = dataset['labels']
        if labels.size == 1:  # Likely that labels is None
            labels = labels.item()
        bin_group = dataset['bin_group']
        min_xy = dataset['min_xy']
        resolution = dataset['resolution']

        pose = self.loader.get_pose(pose_idx)

        txy = pose[:2, 3]  # x, y translation
        base_z = pose[2, 3]  # z location of the base
        bin_size = bin_group.shape[1]
        # grid coordinates
        cx, cy = ((txy - min_xy) / resolution).astype(np.int32)

        elevation_grid, mask = make_grid_points(
            self.width, self.height, points, labels,
            bin_group, self.unique_bin_map, cx, cy, base_z, bin_size, ignore_labels)

        if mode == 'min':
            vis_data = np.min(elevation_grid, axis=2)
        elif mode == 'max':
            vis_data = np.max(elevation_grid, axis=2)
        elif mode == 'mean':
            vis_data = np.mean(elevation_grid, axis=2)
        else:
            raise ValueError('Unknown mode:', mode)

        self.vis.draw_mesh_grid(vis_data, mask, resolution)
        self.vis.set_arrow_pose(pose)
        img = self.vis.render()[:, :, :3]
        return img


def downsample_points(points, resolution, max_num_points_per_cell, rng: np.random.RandomState):
    """
    Args:
        points:
        resolution:
        max_num_points_per_cell:
        rng:

    Returns:
        an array of point indices.
    """
    min_xy, unique_bins, bin_idxs, bin_group = partition_points(
        points, resolution)
    out = []
    for i in range(len(bin_group)):
        if len(bin_group[i]) <= max_num_points_per_cell:
            idxs = bin_group[i]
        else:
            idxs = rng.choice(
                bin_group[i], max_num_points_per_cell, replace=False)
        out.append(idxs)
    return np.concatenate(out)

# def bootstrap(arr, n, rng: np.random.RandomState):
#     rng.choice(arr, n, replace=True)
#     out = []
#     for i in range(n):
#         out.append(arr[rng.randint(len(arr))])
    return out


def torch_unique_2d(arr, device='cuda'):
    """
    This is the 2D version of np.unique() and it runs on GPU.
    Args:
        arr: a Nx2 integer array

    Returns:
        unique_bins: a Mx2 array containing the unique values in @param arr
        bin_idxs: a Nx2 array containing the index of element in @arr in unique_bins
    """

    def make_bins_1d():
        cu_bins = torch.as_tensor(arr, dtype=torch.int32, device=device)
        max_xy, _ = torch.max(cu_bins, dim=0)
        cu_bins_1d = cu_bins[:, 0] * max_xy[1] + cu_bins[:, 1]
        return max_xy.cpu().numpy(), cu_bins_1d

    def find_unique(cu_bins_1d):
        cu_unique_bins_1d, cu_bin_idxs = torch.unique(
            cu_bins_1d, return_inverse=True)
        return cu_unique_bins_1d, cu_bin_idxs.cpu().numpy().astype(np.int32)

    with torch.no_grad():
        max_xy, cu_bins_1d = make_bins_1d()
        cu_unique_bins_1d, bin_idxs = find_unique(cu_bins_1d)
        cu_unique_bins = torch.stack(
            [cu_unique_bins_1d // max_xy[1], cu_unique_bins_1d % max_xy[1]], dim=-1)
        unique_bins = cu_unique_bins.cpu().numpy()

    torch.cuda.empty_cache()
    return unique_bins, bin_idxs


def partition_points(points, resolution):
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    min_xy = np.array([min_x, min_y], np.float32)
    bins = ((points[:, :2] - min_xy) / resolution).astype(np.int32)

    start_time = time.time()
    unique_bins, bin_idxs = torch_unique_2d(bins, device='cpu')
    print('unique time:', time.time() - start_time)
    assert len(bin_idxs) == len(points)

    # bin_idxs contains the bin index of each point
    # group bin_idxs such that bin_group[i] contains the indices of points with bin_idx i
    # to obtain the actual grid x, y location of a bin, use unique_bins[i]
    lookup = np.argsort(bin_idxs)
    sorted_bin_idxs = bin_idxs[lookup]
    boundaries = np.nonzero(np.diff(sorted_bin_idxs) > 0)[0]

    bin_group = List()
    last_b = 0
    for b in boundaries:
        bin_group.append(lookup[last_b: b + 1])
        last_b = b + 1
    bin_group.append(lookup[last_b:])

    return min_xy, unique_bins, bin_idxs, bin_group
