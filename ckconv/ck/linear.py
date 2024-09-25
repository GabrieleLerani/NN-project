import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        if dim == 1:
            self.layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        elif dim == 2:
            self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        elif dim == 3:
            self.layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        else:
            raise ValueError(f"Invalid dimension {dim}. Supported dimensions are 1, 2, and 3.")

    def forward(self, x):
        return self.layer(x)

