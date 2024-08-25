import torch

def linspace_grid(grid_sizes):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    tensors = []
    for size in grid_sizes:
        tensors.append(torch.linspace(-1, 1, steps=int(size)))

    res = torch.meshgrid(*tensors)
    grid = torch.stack(res, dim=0)

    return grid