from torch import nn
from models import S4Block
from ckconv import SepFlexConv
from ckconv.nn.ck import GetLinear
from utils import GetBatchNormalization
from utils import GetAdaptiveAvgPool

class CCNN(nn.Module):
    """
    CCNN architecture (Romero et al., 2022) as defined in the original paper.

          input
            |
        SepFlexConv
            |
        BatchNorm
            |
           GELU
            |
        L x S4Block
            |
        BatchNorm
            |
        GlobalAvgPool
            |
        PointwiseLinear
            |
          output
    """
    def __init__(self, no_blocks, in_channels, out_channels, data_dim):
        """
        Method to init the General Purpose Convolutional Neural Network.
        The model is used to perform spatial data classification.
        The network can be constructed with different block number and channel number.
        """
        super(CCNN, self).__init__()
        
        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SepFlexConv(
            in_channels=in_channels, out_channels=out_channels
        )

        # number of features = the number of output channels of the separable flexible convolutional layer
        self.batch_norm_layer_1 = GetBatchNormalization(data_dim=data_dim, num_features=out_channels)

        self.gelu_layer = nn.GELU()
        
        self.blocks = []
        for i in range(no_blocks):
            # TODO parameters for S4Block
            s4 = S4Block(in_channels=out_channels, out_channels=out_channels, data_dim=data_dim)
            self.blocks.append(s4)

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
