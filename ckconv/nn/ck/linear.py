import torch
import torch.nn as nn

"""
Copied from original repo
"""

def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear2d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear3d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def GetLinear(
    dim: int,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True
) -> torch.nn.Module:
    """
    Returns the Linear Layer.
    """
    if dim == 1:
        return Linear1d(in_channels, out_channels, stride, bias)
    elif dim == 2:
        return Linear2d(in_channels, out_channels, stride, bias)
    elif dim == 3:
        return Linear3d(in_channels, out_channels, stride, bias)
    else:
        raise ValueError(f"Invalid dimension {dim}.")