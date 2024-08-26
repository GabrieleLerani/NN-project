from torch import nn

def GetBatchNormalization(data_dim, num_features):
    if data_dim == 1:
        return nn.BatchNorm1d(num_features)
    elif data_dim == 2:
        return nn.BatchNorm2d(num_features)
    elif data_dim == 3:
        return nn.BatchNorm3d(num_features)