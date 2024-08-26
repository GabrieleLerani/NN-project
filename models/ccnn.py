from torch import nn
from models.modules import S4Block
from ckconv.nn import SepFlexConv
from ckconv.nn.ck import GetLinear
from models.modules.utils import GetBatchNormalization
from models.modules.utils import GetAdaptiveAvgPool
from omegaconf import OmegaConf

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
    def __init__(
        self,  
        in_channels: int,
        out_channels: int, 
        data_dim: int, 
        cfg: OmegaConf
        ):
        """
        Method to init the General Purpose Convolutional Neural Network.
        The model is used to perform spatial data classification.
        The network can be constructed with different block number and channel number.
        """
        super(CCNN, self).__init__()
        
        self.no_blocks = cfg.net.no_blocks

        hidden_channels = cfg.net.hidden_channels

        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SepFlexConv(
            data_dim=data_dim,
            in_channels=in_channels, 
            out_channels=hidden_channels,
            net_cfg=cfg.net,
            kernel_cfg=cfg.kernel
        )

        # number of features = the number of output channels of the separable flexible convolutional layer
        self.batch_norm_layer_1 = GetBatchNormalization(data_dim=data_dim, num_features=hidden_channels)

        self.gelu_layer = nn.GELU()
        
        self.blocks = []
        for i in range(self.no_blocks):
            # TODO parameters for S4Block
            s4 = S4Block(in_channels=hidden_channels, out_channels=hidden_channels, data_dim=data_dim, net_cfg=cfg.net, kernel_cfg=cfg.kernel)
            self.blocks.append(s4)

        self.batch_norm_layer_2 = GetBatchNormalization(data_dim=data_dim, num_features=hidden_channels)

        # global average pooling layer
        # the information of each channel is compressed into a single value
        self.global_avg_pool_layer = GetAdaptiveAvgPool(data_dim=data_dim, output_size=(1,) * data_dim)

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = GetLinear(data_dim, hidden_channels, out_channels)

    def forward(self, x):
        """
        TODO
        """
        out = self.gelu_layer(
            self.batch_norm_layer_1(
                self.sep_flex_conv_layer(x)
            )
        )

        for i in range(self.no_blocks):
            out = self.blocks[i](out)
        
        out = self.pointwise_linear_layer(self.global_avg_pool_layer(self.batch_norm_layer_2(out)))

        return out.squeeze()
