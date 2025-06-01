import torch
import torch.nn as nn

import torch_scatter

from creste.models.blocks.conv import ConvEncoder
from creste.utils.visualization import (numpy_to_pcd, show_bev_map)

DEBUG_SPLAT=0
SAVE_VISUALS=False

class Camera2World(nn.Module): 
    """
    This class convert 2d pixels to 3d points
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Projects an image to a points in the LiDAR coordinate frame.
        Then collapses points in the z direction to get a BEV image.
        
        Inputs:
            depth - [B, N, H, W] Depth image in temporal order
            p2p - [B, N, 4, 4] Camera to LiDAR frame projection matrix (homogeneous)
        Outputs:
            xyz - [B, N, 3, H, W] Points in LiDAR frame
        """
        depth, p2p = x
        B, N, H, W = depth.shape

        depth   = depth.view(B*N, 1, H, W) # [BN, H, W]
        p2p     = p2p.view(B*N, 4, 4) # [BN, 4, 4]

        # TODO: Investigate if positional encoding is needed to correct for errors in depth
        u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        campts = torch.tile(
            torch.stack([
                u, v, torch.ones_like(u)
            ], dim=0), (B*N, 1, 1, 1)
        ).to(depth.device) # B x 3 x H x W
        campts = campts * depth # Convert [u, v 1, 1/d] -> [ud, vd, d, 1]
        campts = torch.cat([campts, torch.ones_like(depth)], dim=1) # [B, 3, H, W]

        # concatenate the points with 1s
        _, _, H, W = campts.shape
        xyz = torch.bmm(p2p, campts.flatten(start_dim=2)) # [B, 4, 3]*[B, 3, H, W]
        xyz = xyz.view(B, N, 4, H, W)[:, :, :3, :, :] # [B, N, 3, H, W]

        return xyz

class Camera2MapMulti(nn.Module):
    """
    This class projects RGB features onto a point cloud
    """
    def __init__(
            self,
            model_cfg, 
            mode="bilinear",
            scatter_mode="mean"    
        ):
        super().__init__()
        self.model_cfg          = model_cfg

        # Register parameters
        self.register_buffer("point_cloud_range", 
                             torch.tensor(model_cfg.point_cloud_range)
        )
        self.register_buffer("max_bound", 
            self.point_cloud_range[3:].reshape(1, -1)
        )
        self.register_buffer("min_bound", 
            self.point_cloud_range[:3].reshape(1, -1)
        )
        self.register_buffer("voxel_size", torch.tensor(model_cfg.voxel_size))
        self.register_buffer("grid_size", 
            ((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).long()
        )

        self.register_buffer("lidar2map",
            torch.tensor([
                [0, -1, 0, -self.min_bound[0, 0]],
                [-1, 0, 0, -self.min_bound[0, 1]],
                [0, 0, -1, -self.min_bound[0, 2]],
                [0, 0, 0, 1]
            ]).float()
        ) # [1, 4, 4] LiDAR to map frame transformation matrix

        self.mode           = mode
        self.scatter_mode   = scatter_mode
        self.min_weight     = 1.0
        self.NC             = model_cfg.get("num_cams", 2)

        self.cam2world = Camera2World()

        # Elevation feature
        if model_cfg['z_embed_mode']=='mlp':
            self.z_proj = nn.Sequential(
                nn.Linear(1, model_cfg['z_embed_dim']*2, bias=True),
                nn.ReLU(),
                nn.Linear(model_cfg['z_embed_dim']*2, model_cfg['z_embed_dim'], bias=True),
                nn.ReLU()
            )
        else:
            raise Exception("Unknown z_embed_mode:", model_cfg['z_embed_mode'])

        # Vision elevation fusion
        self.vision_fusion = ConvEncoder(model_cfg.vision_fusion)

    def _world_to_ref(self, x):
        """
        Transforms points from LiDAR frame to orientation stable BEV frame
        Inputs:
            xyz - [B, N, 3, H, W] Points in LiDAR frame
            ego2ref - [B, 4, 4] Pose matrix relative to first frame in sequence
        Outputs:
            y - [B, N, 3, H, W] Points in orientation stable BEV frame
        """
        xyz, ego2ref = x
        B, N, C, H, W = xyz.shape

        xyz = xyz.permute(0, 1, 3, 4, 2).view(B*N, H*W, C) # [BN, HW, 3]
        xyz = torch.cat([xyz, torch.ones((B*N, H*W, 1), device=xyz.device)], dim=2) # [BN, HW, 4]
        ego2ref = ego2ref.view(B*N, 4, 4) # [BN, 4, 4]

        xyz = torch.bmm(xyz, ego2ref)[:, :, :3] # [BN, H*, 4]*[BN, 4, 4] -> [BN, HW, 4] -> [BN, HW, 3]

        return xyz.permute(0, 2, 1).view(B, N, C, H, W)
    
    def _prepare_features_and_coords(self, x):
        """
        Inputs:
            depth - [B, N, H, W] Depth image in temporal order
            p2p - [B, N, 4, 3] Camera to LiDAR frame projection matrix (homogeneous)
            ego2ref - [B, 4, 4] Pose matrix relative to first frame in sequence
        Outputs:
            xyz - [B, N, 3, H, W] Points in orientation stable LiDAR frame
            xyz_mask - [B, N, 1, H, W] Mask of valid points
            feats - [B, N, F, H, W] RGB features in temporal order
        """
        depth, feats, p2p = x
        B, N, F, H, W = feats.shape
 
        #1 Project RGB features to LiDAR frame
        xyz = self.cam2world( (depth, p2p) )

        #2 Transform RGB features to orientation stable lidar frame (Deprecate this step)
        # xyz = self._world_to_ref( (xyz, ego2ref) ) # [B, N, 3, H, W]

        #3 Obtain elevation features from z and fuse with vision features
        z = xyz[:, :, 2, :, :].unsqueeze(2).permute(
            0, 1, 3, 4, 2
        ).reshape(B*N*H*W, 1) # [B, N, H, W] -> [BNHW, 1]
        
        z_feats = self.z_proj(z) # [BNHW, 1] -> [BNHW, z_embed_dim]
        z_feats = z_feats.view(B, N, H, W, -1).permute(0, 1, 4, 2, 3)
        feats = torch.cat([feats, z_feats], dim=2) # [B, N, F+z_embed_dim, H, W]

        #4 Reduce dimension of concatenated features for map fusion
        feats = self.vision_fusion(
            feats.view(B*N, -1, H, W)
        ) # [BN, F+z_embed_dim, H, W] -> [BN, C, H, W]
        C = feats.shape[1]
        feats = feats.view(B, N, C, H, W)
        
        #5 Obtain xyz_mask for valid points
        xyz = xyz.permute(0, 1, 3, 4, 2).view(B*N, H*W, 3) # [B, N, H, W, 3] -> [B, N*H*W, 3]
        xyz_mask = torch.all((xyz < self.max_bound) & (xyz >= self.min_bound), dim=2, keepdim=True)
        xyz_mask = xyz_mask.view(B, N, H, W, 1).permute(0, 1, 4, 2, 3) # [B, N*H*W, 1] -> [B, N, 1, H, W]
        xyz = xyz.view(B, N, H, W, 3).permute(0, 1, 4, 2, 3) # [B, N*H*W, 3] -> [B, N, 3, H, W]

        return xyz, xyz_mask, feats
    
    def _points_to_voxels(self, points):
        """
        Compute the voxel indices for all valid points and features in the point cloud

        Inputs:
            points - [B, NHW, 3] Points in orientation stable LiDAR frame
            feats - [B, NHW, F] RGB features in temporal order
        Outputs:
            voxels - [B, NHW, 2] Voxel indices for all valid points in grid map
        """
        points = torch.cat([points, torch.ones_like(points[:, :, :1])], dim=2)
        points = (self.lidar2map @ points.permute(0, 2, 1)).permute(0, 2, 1)
        voxels = points[:, :, :2] / self.voxel_size[:2]

        return voxels

    def forward(self, x):
        """
        Inputs:
            x[0] - [B, N, H, W] Depth image in temporal order
            x[0] - [B, N, F, H, W] RGB features in temporal order
            x[1] - [B, 4, 3] Camera to LiDAR frame projection matrix (homogeneous)
            x[2] - [B, 4, 4] Pose matrix relative to first frame in sequence
            x[3] = [B, N, H, W] Mask for immovable objects [Optional]
        Outputs:
            y - [BN, C, H, W] RGB features projected onto BEV map
        """
        assert len(x) >=3, "Input must contain depth, features and camera projection matrix."
        
        #1 Obtain terrain features in orientation stable LiDAR frame
        xyz, xyz_mask, feats = self._prepare_features_and_coords(x[:3])

        if SAVE_VISUALS:
            torch.save(xyz, "xyz.pt")
            torch.save(xyz_mask, "xyz_mask.pt")
            torch.save(feats, "feats.pt")
            import pdb; pdb.set_trace()

        ret_suffix=""
        if self.training: # TODO: Decide if torch on grad is needed
            if len(x)==4:
                xyz_mask = xyz_mask * x[3].unsqueeze(2) # [B, N, 1, H, W] * [B, N, 1, H, W] -> [B, N, 1, H, W]
                ret_suffix="_mv"

        feats = feats * xyz_mask # Mask out invalid and movable points
        B, N, F, H, W = feats.shape
        NS = N // self.NC # Number of views per timestep
        assert N % self.NC == 0,f'"Number of frames must be divisible by {self.NC}'
 
        #2 Convert from LiDAR to BEV Map frame
        xyz     = xyz.permute(0, 1, 3, 4, 2).view(B, NS, self.NC, H*W, 3) # [B, N, H, W, 3] -> [B, N/NC, NC, H*W, 3]
   
        # Concatenate xyz features for all cameras
        xyz     = xyz.view(B*NS, self.NC*H*W, 3) # [B, N, NC, H*W, 3] -> [B*N, NC*H*W, 3]
        feats   = feats.permute(0, 1, 3, 4, 2).view(B, NS, self.NC, H*W, F) # [B, N, H, W, F] -> [B, N, NC, H*W, F]
        feats   = feats.view(B, NS, self.NC*H*W, F) # [B, N, NC, H*W, F] -> [B, N, NC*H*W, F]
        feats   = feats.permute(0, 1, 3, 2).view(B*NS, F, self.NC*H*W) # [B, N, 2*H*W, F] -> [B, N, F, 2*H*W]
        xy      =  self._points_to_voxels(xyz)

        #3 Splat LiDAR frame features to BEV map
        if self.mode=="bilinear":
            splat_feats, splat_densities = self.splat_soft((xy, feats, self.grid_size[:2])) # [B, N*H*W, C]
        else:
            raise Exception("Unknown splat mode:", self.mode)

        if DEBUG_SPLAT:
            print("Debugging splat projection")
            import numpy as np
            pc_tensor = xyz[0]
            numpy_to_pcd(pc_tensor.detach().cpu().numpy(), "testxyz.pcd")
            # Sanity check BEV feature map to see if it is roughly correct
            show_bev_map(
                splat_feats.view(B, NS, F, self.grid_size[0], self.grid_size[1]),
                splat_densities.view(B, NS, self.grid_size[0], self.grid_size[1]),
                bev_ids=[0, 1, 2]
            )
            # show_bev_map(splat_feats.view(B, NS, F, self.grid_size[0], self.grid_size[1]), splat_densities.view(B, NS, self.grid_size[0], self.grid_size[1]), bev_idx=2)

        splat_feats = splat_feats.view(B*NS, F, self.grid_size[0], self.grid_size[1])
        splat_densities = splat_densities.view(B*NS, self.grid_size[0], self.grid_size[1], 1).permute(0, 3, 1, 2) # [B*NS, H, W, 1] -> [B*NS, 1, H, W]

        return {
            f'bev_features{ret_suffix}': splat_feats, 
            f'bev_densities{ret_suffix}': splat_densities, 
            f'bev_coords{ret_suffix}': xy
        }

    def splat_soft(self, x):
        """
        Copied from pytorch3d.ops.points_to_volumes
        
        Convert a batch of point clouds to a batch of volumes using trilinear
        splatting into a volume.

        Args:
            points_2d: Batch of 2D point coordinates of shape
                Coordinates have t.
                `(minibatch, N, 2)` where N is the number of points.
            points_features: Features of shape `(minibatch, feature_dim, N)`
                corresponding to the points of the input point cloud `points_2d`.
            grid_size: (H, W) tuple, representing the
                spatial resolutions of each of the the non-flattened `volumes` tensors.
            min_weight: A scalar controlling the lowest possible total per-voxel
                weight used to normalize the features accumulated in a voxel.
            mode: how to aggregate the features. Can be 'mean', 'sum' or 'max'
        Returns:
            volume_features: Output volume of shape `(minibatch, feature_dim, N_voxels).
        """
        points_2d, points_features, grid_size = x

        H, W = grid_size
        n_voxels = torch.prod(grid_size)
        ba, feature_dim, n_points = points_features.shape

        # XY = the upper-left volume index of the 4-neighborhood of every point
        # grid_size is of the form (minibatch, height-width)
        grid_size_xy = grid_size[[1, 0]]

        XY = points_2d.floor().long()
        rXY = points_2d - XY.type_as(points_2d)  # remainder of floor
        
        # split into separate coordinate vectors
        X, Y = XY.split(1, dim=2)

        # rX = remainder after floor = 1-"the weight of each vote into
        #      the X coordinate of the 4-neighborhood"
        rX, rY = rXY.split(1, dim=2)

        # get random indices for the purpose of adding out-of-bounds values
        rand_idx = X.new_zeros(X.shape).random_(0, n_voxels)
        
        volume_densities = points_features.new_zeros(ba, n_voxels, 1)
        volume_features = points_features.new_zeros(ba, points_features.shape[1], n_voxels)
        # iterate over the x, y indices of the 4-neighborhood (xdiff, ydiff)
        for xdiff in (0, 1):
            X_ = X + xdiff
            wX = (1 - xdiff) + (2 * xdiff - 1) * rX
            for ydiff in (0, 1):
                Y_ = Y + ydiff
                wY = (1 - ydiff) + (2 * ydiff - 1) * rY

                # weight of each vote into the given cell of 4-neighborhood
                w = wX * wY

                # valid - binary indicators of votes that fall into the volume
                valid = ((0 <= X_) * (X_ < W) * (0 <= Y_) * (Y_ < H)).long()

                # linearized indices into the volume
                idx = Y_ * W + X_

                # out-of-bounds features added to a random voxel idx with weight=0.
                idx_valid = idx * valid + rand_idx * (1 - valid)
                w_valid = w * valid.type_as(w)

                # scatter add casts the votes into the weight accumulator
                # and the feature accumulator
                volume_densities.scatter_add_(1, idx_valid, w_valid)

                # reshape idx_valid -> (minibatch, feature_dim, n_points)
                idx_valid = idx_valid.view(ba, 1, n_points).expand_as(points_features)
                w_valid = w_valid.view(ba, 1, n_points)

                # volume_features of shape (minibatch, feature_dim, n_voxels)
                if self.scatter_mode == 'mean' or self.scatter_mode == 'sum':
                    volume_features.scatter_add_(2, idx_valid, w_valid * points_features)
                elif self.scatter_mode == 'max':
                    _ = torch_scatter.scatter(
                        points_features * w_valid, idx_valid,
                        dim=2, reduce='max', dim_size=volume_features.shape[2])
                    volume_features = torch.maximum(_, volume_features)
                else:
                    raise Exception('Unknown splat scatter mode:', self.scatter_mode)

        if self.scatter_mode == 'mean':
            # Divide each feature by the total weight of the votes
            volume_features = volume_features / volume_densities.view(ba, 1, n_voxels).clamp(
                self.min_weight
            )

        return volume_features, volume_densities
