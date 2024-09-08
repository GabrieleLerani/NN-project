from torch import nn

def GetDropout(data_dim: int, p: float) -> nn.Module:
    if data_dim == 1:
        return nn.Dropout1d(p = p)
    elif data_dim == 2:
        return nn.Dropout2d(p = p)
    elif data_dim == 3:
        return nn.Dropout3d(p = p)