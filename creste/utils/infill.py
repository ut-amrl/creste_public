import torch
from torch.nn import functional as F
import numpy as np

def construct_idw_kernel(window_size):
    center = window_size // 2
    xrange = torch.arange(window_size, device="cuda")
    yrange = torch.arange(window_size, device="cuda")
    xv, yv = torch.meshgrid(xrange, yrange, indexing='ij')
    distances = torch.sqrt((xv - center)**2 + (yv - center)**2)

    # Apply inverse distance weighting
    kernel = 1 / (distances + 1)  # Adding 1 to avoid division by zero at the center
    kernel[center, center] = 0  # Optionally, set the center to 0 if you don't want to include the central pixel
    kernel /= kernel.sum()

    return kernel

def idw_infill(depth_image, window_size=5):
    """
    Inputs:
    depth_image [H, W] (np.ndarray) - Depth image to be infilled
    window_size (int) - radius of window to use for IDW, must be odd
    """
    depth_image = torch.from_numpy(depth_image).cuda().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    depth_mask = depth_image > 0

    #1 Construct weighting kernel
    kernel = construct_idw_kernel(window_size)
    kernel = kernel.unsqueeze(0).unsqueeze(0) # [1, 1, window_size, window_size]
    
    dense_depth = F.conv2d(depth_image, kernel, padding=window_size//2)

    # Correct for original depths
    dense_depth[depth_mask] = depth_image[depth_mask]

    return dense_depth.squeeze(0).squeeze(0).cpu().numpy()

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    mX = np.zeros((m,n)) + float("inf")
    mY = np.zeros((m,n)) + float("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.int32(Pts[0])#np.round(Pts[0])  # NOTICE
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.int32(Pts[1])#np.round(Pts[1])  # NOTICE
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] + i - grid - 1  # NOTICE
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] + j - grid - 1  # NOTICE
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])

    # Inverse distance weighted
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s

    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/(S+1e-12)
    return out