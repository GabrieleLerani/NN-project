from torch import nn
from models import S4Block
from ckconv import SepFlexConv
from ckconv.nn.ck import GetLinear
from utils import GetBatchNormalization
from utils import GetAdaptiveAvgPool

class CCNN(nn.Module):
    def __init__(self, L, in_channels, out_channels, data_dim):
        """
        Method to init the General Purpose Convolutional Neural Network.
        The model is used to perform spatial data classification.
        The network can be constructed with different block number and channel number.
        """
        super(CCNN, self).__init__()
        # TODO parameters of the model
        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SepFlexConv(
            in_channels=in_channels, out_channels=out_channels
        )

        # number of features = the number of output channels of the separable flexible convolutional layer
        self.batch_norm_layer_1 = GetBatchNormalization(data_dim=data_dim, num_features=out_channels)

        self.gelu_layer = nn.GELU()
        
        self.blocks = []
        for i in range(L):
            # TODO parameters for S4Block
            self.blocks.append(S4Block())

        self.batch_norm_layer_2 = GetBatchNormalization(data_dim=data_dim, num_features=out_channels)

        # global average pooling layer
        # the information of each channel is compressed into a single value
        self.global_avg_pool_layer = GetAdaptiveAvgPool(data_dim=data_dim, output_size=(1,) * data_dim)

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = GetLinear(data_dim, in_channels, out_channels)

    def forward(self, x):
        """
        TODO
        """
        out = self.gelu_layer(
            self.batch_norm_layer_1(
                self.sep_flex_conv_layer(x)
            )
        )

        for i in range(len(self.blocks.size)):
            out = self.blocks[i](out)
        
        out = self.pointwise_linear_layer(self.global_avg_pool_layer(self.batch_norm_layer_2(out)))

        return out
