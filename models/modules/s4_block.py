from torch import nn
from ckconv import SepFlexConv
from ckconv.ck import LinearLayer
from models.modules.utils import GetBatchNormalization
from models.modules.utils import GetDropout
from omegaconf import OmegaConf

class S4Block(nn.Module):
    """
    Create a S4 block (Gu et al., 2022) as defined in the Continuous CNN architecture.

          input
            |
    | -------------|
    |            BarchNorm             
    |            SepFlexConv             
    |            GELU     
    |            DropOut          
    |            PointwiseLinear                
    |            GELU             
    |              |
    |---->(+)<-----|
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
            dropout: float 
        ):
        """
        Method to init the S4 block
        """
        super().__init__()

        self.batch_norm_layer = GetBatchNormalization(data_dim=data_dim, num_features=in_channels)

        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SepFlexConv(
            data_dim=data_dim,
            in_channels=in_channels,
            net_cfg=net_cfg,
            kernel_cfg=kernel_cfg
        )
        
        self.gelu_layer = [nn.GELU(), nn.GELU()]

        self.dropout_layer = GetDropout(data_dim=data_dim, p=dropout)

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = LinearLayer(data_dim, in_channels, out_channels)

        self.seq_modules = nn.Sequential(
            self.batch_norm_layer,
            self.sep_flex_conv_layer,
            self.gelu_layer[0],
            self.dropout_layer,
            self.pointwise_linear_layer,
            self.gelu_layer[1]
        )

        # init last linear layer
        nn.init.kaiming_normal_(self.pointwise_linear_layer.layer.weight)
        self.pointwise_linear_layer.layer.bias.data.fill_(0.0)
        
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
