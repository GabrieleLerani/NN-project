from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    padding: int,
    groups: int,
):
    return torch.nn.functional.conv1d(x, kernel, bias=bias, padding=padding, stride=1, groups=groups)

def conv2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    padding: int,
    groups: int,
):
    return torch.nn.functional.conv2d(x, kernel, bias=bias, padding=padding, stride=1, groups=groups)

def conv3d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    padding: int,
    groups: int,
):
    return torch.nn.functional.conv3d(x, kernel, bias=bias, padding=padding, stride=1, groups=groups)


def get_conv_function(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    padding: int,
    groups: int,
    dim: int,
):
    """
    Returns the Convolutional Layer.
    """
    if dim == 1:
        return conv1d(x, kernel, bias, padding, groups)
    elif dim == 2:
        return conv2d(x, kernel, bias, padding, groups)
    elif dim == 3:
        return conv3d(x, kernel, bias, padding, groups)
    else:
        raise ValueError(f"Invalid dimension: {dim}")


def conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    causal: bool = False,
):

    data_dim = len(x.shape) - 2
    # -> [batch_size, channels, x_dimension, y_dimension, ...] -> len[x.shape] = 2 + data_dim

    kernel_size = torch.tensor(kernel.shape[-data_dim:])
    assert torch.all(
        kernel_size % 2 != 0
    ), f"Convolutional kernels must have odd dimensionality. Received {kernel.shape}"
    # Pad by kernel_size // 2 so that the output has the same size as the input
    padding = (kernel_size // 2).tolist()

    groups = kernel.shape[1]

    if causal:
        # Pad the input to the left
        x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)

    # Invert first two dimensions of kernel because there should be one kernel per input channel
    kernel = kernel.view(kernel.shape[1], 1, *kernel.shape[2:])

    return get_conv_function(x, kernel, bias, padding=0 if causal else padding, groups=groups, dim=data_dim)


def fftconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    data_dim = len(x.shape) - 2
    # -> [batch_size, channels, x_dimension, y_dimension, ...] -> len[x.shape] = 2 + data_dim

    assert data_dim == 1

    # Pad the input to the left
    x_padded = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)

    if x_padded.shape[-1] % 2 == 0:
        x_padded = F.pad(x_padded, [1, 0])

    # Padding kernel
    kernel_padded = F.pad(kernel, [0, x_padded.size(-1) - kernel.size(-1)])

    # Fourier Transform
    x_fr = torch.fft.rfft(x_padded, dim=-1)

    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    kernel_fr = torch.conj(torch.fft.rfft(kernel_padded, dim=-1))

    # Element-wise Multiplication in Fourier domain
    output_fr = x_fr * kernel_fr

    # This part of the code ensures that the output tensor out has the same spatial dimensions as the original input tensor x (before padding)
    out = torch.fft.irfft(output_fr, dim=-1)[..., : x.shape[-1]]

    if bias is not None:
        out = out + bias.view(1, -1, *([1] * data_dim))

    return out