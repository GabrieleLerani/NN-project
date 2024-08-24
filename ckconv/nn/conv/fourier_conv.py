import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def padding():
  pass

# TODO check dimensions
def fftconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    separable: bool = False,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    
    data_dim = len(x.shape) - 2
    # -> [batch_size, channels, x_dimension, y_dimension, ...] -> len[x.shape] = 2 + data_dim
    
    # check if the kernel is odd
    print(kernel.shape[-data_dim:])
    kernel_size = torch.tensor(kernel.shape[-data_dim:]) 
    assert torch.all(
        kernel_size % 2 != 0
    ), f"Convolutional kernels must have odd dimensionality. Received {kernel.shape}"
  
    # padding input
    padding_x = kernel.shape[-1] // 2
    padding_x = (2 * data_dim) * [
        padding_x
    ]

    x_padded = F.pad(x, padding_x, mode='constant', value=0)

    if x.shape[-1] % 2 != 0:
        x_padded = F.pad(x_padded, [0, 1])

    # padding kernel
    padding_kernel = [
        pad
        for i in reversed(range(2, x.ndim))
        for pad in [0, x.shape[i] - kernel.shape[i]]
    ]

    kernel_padded = F.pad(kernel, padding_kernel, mode='constant', value=0)
    
    # Fourier Transform
    x_fr = torch.fft.rfftn(x_padded, dim=tuple(range(2, x.ndim)))
    kernel_fr = torch.fft.rfftn(kernel_padded, dim=tuple(range(2, kernel.ndim)))
    
    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    kernel_fr = torch.conj(kernel_fr)
    output_fr = torch.einsum("bi..., oi... -> bo...", x_fr, kernel_fr)

    # Element-wise Multiplication in Fourier domain
    out = torch.fft.irfftn(output_fr, dim=tuple(range(2, x.ndim))).float()
    
    # Generalization to higher dimensions
    slices = [slice(None), slice(None)] 
    slices.extend(slice(None, x.shape[-i]) for i in range(1, data_dim + 1))
    out = out[tuple(slices)]
    
    # Add bias if provided
    if bias is not None:
        out += bias.view(1, -1, *([1] * data_dim))
    
    return out



batch_size = 2
num_channels = 3

x = torch.randn(batch_size, num_channels, 16)
kernel = torch.randn(num_channels, num_channels, 3)
bias = torch.randn(num_channels)
output = fftconv(x, kernel, bias)

print(output)


