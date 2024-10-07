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
    causal: bool = False,
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

    if causal:
        # pad the input to the left
        x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)

    # invert first two dimensions of kernel because there should be one kernel per input channel
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
        f"expected_shape : [batch_size, num_channels, x_dim1, x_dim2, ...] , x_fr shape: {x_fr.shape}"
    )
    print(
        f"expected_shape : [num_channels, k_dim1, k_dim2, ...] ,kernel_fr shape: {kernel_fr.shape}"
    )
    # Element-wise Multiplication in Fourier domain
    output_fr = x_fr * kernel_fr

    # Inverse FFT to transform the result back to the spatial domain
    out = torch.fft.irfftn(output_fr, dim=tuple(range(2, x_padded.ndim))).float()

    # This part of the code ensures that the output tensor out has the same spatial dimensions as the original input tensor x (before padding)

    # Select all elements in the batch_size and channels dimensions (first two dimensions of out)
    slices = [slice(None), slice(None)]
    # Extension of the slices list to include slices for each spatial dimension (for all dimensions [data_dim])
    # Let's assume x_padded has a shape of [batch_size, channels, height, width]. After this step, slices might look like:
    # - slices = [slice(None), slice(None), slice(None, height), slice(None, width)]
    slices.extend(slice(None, x.shape[-i]) for i in range(1, data_dim + 1))
    # This operation effectively crops the out tensor to remove any padding that was added during the earlier steps
    out = out[tuple(slices)]

    # Add bias if provided
    if bias is not None:
        out = out + bias.view(1, -1, *([1] * data_dim))

    return out




def fftconv1d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    data_dim = len(x.shape) - 2
    assert data_dim == 1

    kernel_size = torch.tensor(kernel.shape[-data_dim:])

    x_shape = x.shape

    # 1. Handle padding
    # pad the input to the left
    x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)

    # 2. Pad the kernel tensor to make them equally big. Required for fft.
    kernel = F.pad(kernel, [0, x.size(-1) - kernel.size(-1)])

    # 3. Perform fourier transform
    x_fr = torch.fft.rfft(x, dim=-1)
    kernel_fr = torch.conj(torch.fft.rfft(kernel, dim=-1))

    # 4. Multiply the transformed matrices:
    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    output_fr = kernel_fr * x_fr

    # 5. Compute inverse FFT, and remove extra padded values
    # Once we are back in the spatial domain, we can go back to float precision, if double used.
    out = torch.fft.irfft(output_fr, dim=-1)[..., : x_shape[-1]]

    # 6. Optionally, add a bias term before returning.
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out