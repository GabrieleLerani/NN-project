from torch import nn
from ckconv.nn import SepFlexConv
from ckconv.nn.ck import LinearLayer
from models.modules.utils import GetBatchNormalization
from models.modules.utils import GetDropout
from omegaconf import OmegaConf

class TCNBlock(nn.Module):
    """
    Create a TCN block ( Bai et al., 2017 ) as defined in Romero et al., 2022
    where standard conv is replaced by SepFlexConv

    input
     | ---------------|
     SepFlexConv      |
     Norm             |
     NonLinearity     |
     DropOut          |
     |                |
     SepFlexConv      |
     Norm             |
     NonLinearity     |
     DropOut          |
     + <--------------|
     |
     NonLinearity
     |
     output
    """


    def __init__(
            self,
            in_channels,
            out_channels,
            data_dim,
            net_cfg: OmegaConf,
            kernel_cfg: OmegaConf, 
        ):
        """
        Method to init the S4 block
        """
        super().__init__()

        self.batch_norm_layer = [
            GetBatchNormalization(data_dim=data_dim, num_features=in_channels),
            GetBatchNormalization(data_dim=data_dim, num_features=in_channels)
        ]

        # separable flexible convolutional layer
        self.sep_flex_conv_layer = [

            SepFlexConv(
                data_dim=data_dim,
                in_channels=in_channels,
                net_cfg=net_cfg,
                kernel_cfg=kernel_cfg
            ),
            SepFlexConv(
                data_dim=data_dim,
                in_channels=in_channels,
                net_cfg=net_cfg,
                kernel_cfg=kernel_cfg
            )
        ]

        
        self.gelu_layer = [nn.GELU(), nn.GELU()]

        self.dropout_layer = [GetDropout(data_dim=data_dim),GetDropout(data_dim=data_dim)]

        self.seq_modules = nn.Sequential(
            
            self.sep_flex_conv_layer[0],
            self.batch_norm_layer[0],
            self.gelu_layer[0],
            self.dropout_layer[0],
            self.sep_flex_conv_layer[1],
            self.pointwise_linear_layer,
            self.gelu_layer[1],
            self.dropout_layer[1],
        )

        # Used in residual networks (ResNets) to add a direct path from the input to the output, 
        # which helps in training deeper networks by mitigating the vanishing gradient problem.
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(LinearLayer(data_dim, in_channels, out_channels))
            nn.init.kaiming_normal_(shortcut[0].weight)
            if shortcut[0].bias is not None:
                shortcut[0].bias.data.fill_(value=0.0)
        # If no layer is added (because in_channels and out_channels were the same), 
        # the shortcut will be empty and effectively be an identity mapping.
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        """
        Standard method of nn.modules we embed also the residual connection
        """
        shortcut = self.shortcut(x)
        out = self.seq_modules(x)
        return out + shortcut