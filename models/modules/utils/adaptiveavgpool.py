from torch import nn

def GetAdaptiveAvgPool(data_dim, output_size):
    if data_dim == 1:
        return nn.AdaptiveAvgPool1d(output_size)
    elif data_dim == 2:
        return nn.AdaptiveAvgPool2d(output_size)
    elif data_dim == 3:
        return nn.AdaptiveAvgPool3d(output_size)