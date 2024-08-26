from torch import nn
from ckconv.nn import SepFlexConv
from ck import GetLinear
from utils import GetBatchNormalization
from utils import GetDropout

class S4Block(nn.Module):
    """
    Create a S4 block (Gu et al., 2022) as defined in the Continuous CNN architecture.

          input
            |
    | -------------|
    |            BarchNorm             
    |            SepFlecConv             
    |            GELU     
    |            DropOut          
    |            PointwiseLinear                
    |            GELU             
    |              |
    |---->(+)<-----|
           |
        output

    """


    def __init__(self, in_channels, out_channels, data_dim):
        """
        Method to init the S4 block
        """
        super().__init__()

        self.batch_norm_layer = GetBatchNormalization(data_dim=data_dim, num_features=in_channels)

        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SepFlexConv(
            data_dim=data_dim,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_no_layers=kernel_no_layers,
            kernel_hidden_channels=kernel_hidden_channels,
            kernel_size=kernel_size,
            conv_type=conv_type,
            fft_thresold=fft_thresold,
            bias=bias
        )

        self.gelu_layer1 = nn.GELU()

        self.dropout_layer = GetDropout(data_dim=data_dim)

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = GetLinear(data_dim, in_channels, out_channels)

        self.gelu_layer2 = nn.GELU()

        # Used in residual networks (ResNets) to add a direct path from the input to the output, 
        # which helps in training deeper networks by mitigating the vanishing gradient problem.
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(GetLinear(data_dim, in_channels, out_channels))
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

        out = self.gelu_layer2(
            self.pointwise_linear_layer(
                self.dropout_layer(
                    self.gelu_layer1(
                        self.sep_flex_conv_layer(
                            self.batch_norm_layer(x)
                        )
                    )
                )
            )
        )

        return out + shortcut
