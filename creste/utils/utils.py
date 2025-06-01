import torch
import torch.nn.functional as F
import kornia.geometry.transform as T
import numpy as np

def warp(input_tensor, transform, interpolation,
         precision=None, output_size=None, padding_mode='zeros'):
    if precision is None:
        precision = transform.dtype
    else:
        transform = transform.to(precision)

    inp = input_tensor.to(precision)

    # B, C, H, W
    assert(input_tensor.ndim == 4)

    # B, 2, 3
    assert(transform.ndim == 3)

    # add a mask channel
    inp_plus_mask = F.pad(inp,
                       (0, 0, 0, 0, 0, 1),
                       value=1.0)
    if output_size is None:
        output_size = inp_plus_mask.shape[-2:]
    inp_plus_mask = T.warp_affine(inp_plus_mask,
                                  transform,
                                  dsize=output_size,
                                  align_corners=False,
                                  mode=interpolation,
                                  padding_mode=padding_mode)

    # extract output and mask
    output = inp_plus_mask[:, :-1].to(input_tensor.dtype)
    mask = (inp_plus_mask[:, -1] > 0.99) #.to(torch.bool)

    return output, mask

def make_labels_contiguous_vectorized(labels):
    """
    Convert a tensor of shape 1xHxW containing label indices to a tensor of the same shape
    with contiguous labels starting from 0, using a vectorized approach.
    
    Args:
    labels (torch.Tensor): Input tensor of shape 1xHxW where each value is a label index. [1, H, W]

    Returns:
    torch.Tensor: Tensor of the same shape with contiguous label indices.
    """
    # Flatten the tensor to get all labels as a 1D array
    unique_labels, new_label_indices = torch.unique(labels, sorted=True, return_inverse=True)
    
    # Reshape the new_label_indices back to the original shape of the labels tensor
    new_labels = new_label_indices.reshape(labels.shape)

    return new_labels

def remap_labels_in_batch(gt, ignore_idx=0):
    """
    Remap labels from different batches to different classes in ascending order unless its the ignore_idx

    Inputs:
        gt: (torch.Tensor) [B, H, W] tensor of labels
    """
    B, H, W = gt.shape

    offset_idx = 0
    gt_remap = torch.ones_like(gt) * ignore_idx
    for b in range(B):
        remap_dict = {label: i+offset_idx for i, label in enumerate(torch.unique(gt[b])) if label != ignore_idx}

        for label, remap in remap_dict.items():
            gt_remap[b, gt[b] == label] = remap
        offset_idx += len(remap_dict)

    return gt_remap

def remap_and_sum_channels_torch(tensor, mapping):
    """
    Remap and sum the channels of an HxWxC tensor based on a given mapping in PyTorch.
    The mapping is a list where the index is the original channel index, and the value is the new channel index.

    Parameters:
    tensor (torch.Tensor): Input tensor of shape HxWxC.
    mapping (list): A list where the index represents the original channel index and the value is the new index.

    Returns:
    torch.Tensor: The remapped and summed tensor.
    """
    # Convert the mapping list to a tensor
    mapping_tensor = torch.tensor(mapping, dtype=torch.long)

    # Determine the number of new channels
    new_channels = mapping_tensor.max() + 1

    # Create a one-hot encoding of the mapping
    one_hot = torch.nn.functional.one_hot(mapping_tensor, num_classes=new_channels).to(tensor.dtype)

    # Use einsum to multiply and sum in one step
    remapped_tensor = torch.einsum('ijk,kl->ijl', tensor, one_hot)

    return remapped_tensor

def most_frequent_per_index(src, index, num_classes):
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
    num_unique_indices = unique_indices.size(0)

    # Initialize a tensor to count occurrences of each class for each index
    counts = torch.zeros(num_unique_indices, num_classes, dtype=torch.long).to(src.device)
    
    # Use scatter_add to count occurrences of each class at each index
    result =  torch.tile(inverse_indices.unsqueeze(1), (1, num_classes)).to(src.device)
    result.scatter_add_(1, src.unsqueeze(1), torch.ones_like(src.unsqueeze(1), dtype=torch.long))
    counts.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, num_classes), result)

    # Find the most frequent class (argmax over counts)
    most_frequent = torch.argmax(counts, dim=1)

    # Map the results back to the original indices
    result = most_frequent[inverse_indices]
    
    return most_frequent

def drop_overlapping_horizons(finfos, horizon):
    """
    finfos: (np.array) [N,] array of strings with 'seq frame' format. Assumed to 
    be in ascending order by sequence and frame number.
    horizon: (int) Number of frames to consider for horizon. Uses first frame as starting
    """
    # Split sequences and frame numbers
    seqs, frames = np.array([x.split() for x in finfos]).T
    seqs = seqs.astype(int)
    frames = frames.astype(int)

    # Sort by sequence and then by frame number
    sort_idx = np.lexsort((frames, seqs))
    sorted_seqs = seqs[sort_idx]
    sorted_frames = frames[sort_idx]

    # Find where sequences change
    change_points = np.r_[True, sorted_seqs[1:] != sorted_seqs[:-1]]

    # Calculate differences between consecutive frames within the same sequence
    frame_diffs = np.diff(sorted_frames, prepend=sorted_frames[0])

    # Initialize an array to keep track of valid frames
    keep = np.zeros_like(sorted_frames, dtype=bool)

    # Initialize the previous frame index for each sequence
    # prev_frame_idx = np.zeros_like(sorted_seqs, dtype=int)

    # keep = np.logical_or(change_points, frame_diffs >= horizon)
    for i in range(len(sorted_frames)):
        if change_points[i]:  # First frame in a new sequence
            keep[i] = True
        elif sorted_frames[i] - sorted_frames[prev_frame_idx] >= horizon:  # Frame difference >= 50
            keep[i] = True
        else:
            continue
        prev_frame_idx = i  # Update previous frame index for the sequence
    
    # Restore the original order after filtering
    filtered_idx = sort_idx[keep]
    return finfos[filtered_idx]
