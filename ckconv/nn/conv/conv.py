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
        # x = F.pad(x, [padding[0], padding[0]], value=0.0)
        return conv1d(x, kernel, bias, padding, groups)
    elif dim == 2:
        return conv2d(x, kernel, bias, padding, groups)
    elif dim == 3:
        return conv3d(x, kernel, bias, padding, groups)
    else:
        raise ValueError(f"Invalid dimension {dim}")
    

def conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    
    data_dim = len(x.shape) - 2
    # -> [batch_size, channels, x_dimension, y_dimension, ...] -> len[x.shape] = 2 + data_dim

    kernel_size = torch.tensor(kernel.shape[-data_dim:])
    assert torch.all(
        kernel_size % 2 != 0
    ), f"Convolutional kernels must have odd dimensionality. Received {kernel.shape}"
    # pad by kernel_size // 2 so that the output has the same size as the input
    padding = (kernel_size // 2).tolist() 

    groups = kernel.shape[1]
    # invert first two dimensions of kernel because there should be one kernel per input channel
    kernel = kernel.view(kernel.shape[1], 1, *kernel.shape[2:])

    return get_conv_function(x, kernel, bias, padding=padding, groups=groups, dim=data_dim)



def fftconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    data_dim = len(x.shape) - 2
    # -> [batch_size, channels, x_dimension, y_dimension, ...] -> len[x.shape] = 2 + data_dim

    kernel_size = torch.tensor(kernel.shape[-data_dim:])
    assert torch.all(
        kernel_size % 2 != 0
    ), f"Convolutional kernels must have odd dimensionality. Received {kernel.shape}"

    # padding input
    padding_x = kernel.shape[-1] // 2
    padding_x = (2 * data_dim) * [padding_x]

    x_padded = F.pad(x, padding_x)

    if x_padded.shape[-1] % 2 != 0:
        x_padded = F.pad(x_padded, [0, 1])

    # padding kernel
    padding_kernel = [
        pad
        for i in reversed(range(2, x_padded.ndim))
        for pad in [0, x_padded.shape[i] - kernel.shape[i]]
    ]

    kernel_padded = F.pad(kernel, padding_kernel, mode="constant", value=0)

    # Fourier Transform
    x_fr = torch.fft.rfftn(x_padded, dim=tuple(range(2, x_padded.ndim)))
    kernel_fr = torch.fft.rfftn(kernel_padded, dim=tuple(range(2, kernel.ndim)))

    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    # b->batch i->input channel o->output channel , ...> any additional dimensions
    # The einsum notation specifies that for each position in the output tensor,
    # you perform element-wise multiplication of x_fr and kernel_fr and sum over the i dimension (input channels)
    # and any remaining spatial dimensions
    # Essentially, it calculates a form of convolution (correlation) in the Fourier domain,
    # where the i dimension of x_fr is multiplied with the i dimension of kernel_fr
    # and summed to produce the output tensor's o dimension
    kernel_fr = torch.conj(kernel_fr)
    # Assuming x_fr has shape [batch_size, num_channels, x_dim1, x_dim2, ...]
    # and kernel_fr has shape [num_channels, k_dim1, k_dim2, ...]
    print(
        f"expeted_shape : [batch_size, num_channels, x_dim1, x_dim2, ...] , x_fr shape: {x_fr.shape}"
    )
    print(
        f"expeted_shape : [num_channels, k_dim1, k_dim2, ...] ,kernel_fr shape: {kernel_fr.shape}"
    )
    output_fr = torch.einsum("bi..., oi... -> bo...", x_fr, kernel_fr)

    # Element-wise Multiplication in Fourier domain
    out = torch.fft.irfftn(output_fr, dim=tuple(range(2, x_padded.ndim))).float()

    # Generalization to higher dimensions
    slices = [slice(None), slice(None)]
    slices.extend(slice(None, x_padded.shape[-i]) for i in range(1, data_dim + 1))
    out = out[tuple(slices)]

    # Add bias if provided
    if bias is not None:
        out = out + bias.view(1, -1, *([1] * data_dim))

    return out


