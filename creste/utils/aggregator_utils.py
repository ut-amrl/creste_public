"""
Adapted from https://github.com/ut-amrl/object-identification/blob/args/methods/dino/aggregators.py
"""
import torch
import torch_scatter

def aggregate_descriptors(
    ids: torch.Tensor, descriptors: torch.Tensor, dims: tuple, aggregator: str = "GMP"
):
    """
    Aggregates descriptors based on voxel indices and aggregation method

    GMP - Global Max Pooling
    """
    H, W = dims
    F = descriptors.shape[1]
    loc1d = ids[:, 1] * W + ids[:, 0]
    loc1d = loc1d.long()

    if aggregator == "GMP":
        aggregated_descriptors = torch_scatter.scatter(
            descriptors, loc1d, dim=0, dim_size=H * W, reduce="max"
        )
    elif aggregator == "GAP":
        aggregated_descriptors = torch_scatter.scatter(
            descriptors, loc1d, dim=0, dim_size=H * W, reduce="mean"
        )
    else:
        raise ValueError(f"Invalid aggregator {aggregator}")

    aggregated_descriptors = aggregated_descriptors.view((H, W, F))

    return aggregated_descriptors


def get_GeM(descriptor: torch.Tensor, p: float = 3, mode: str = "all"):
    """
    Aggregates descriptors using generalized mean pooling
    Args:
        descriptor (torch.Tensor): [N, N_patches, desc_dim]
        p (float): power of mean
        mode (str): "all" or "real" or "imag" (only for complex descriptor)

    Returns:
        global_descriptor (torch.Tensor): [N, desc_dim] if real else [N, 2 * desc_dim]
    """
    assert len(descriptor.shape) == 3  # [N, N_patches, desc_dim]
    assert p > 0, "p should be positive"

    if p == 1:
        x = torch.mean(descriptor, dim=1)
    elif p == float("inf"):
        x = torch.max(descriptor, dim=1)[0]
    else:
        x = torch.mean(descriptor**p, dim=1)
        x = x.to(torch.complex64) ** (1 / p)
        if mode == "all":
            x = torch.cat([x.real, x.imag], dim=1)
        elif mode == "real":
            x = torch.real(x)
        elif mode == "imag":
            x = torch.imag(x)
    return x