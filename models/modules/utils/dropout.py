from torch import nn

def GetDropout(data_dim):
    if data_dim == 1:
        return nn.Dropout1d()
    elif data_dim == 2:
        return nn.Dropout2d()
    elif data_dim == 3:
        return nn.Dropout3d()