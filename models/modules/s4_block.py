from torch import nn
from models import SFC
from ckconv.nn import SepFlexConv

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


    def __init__(self, in_channels, out_channels):
        """
        Method to init the S4 block
        """
        super().__init__()

        self.batch_norm_layer = nn.BatchNorm2d(num_features=in_channels)

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

        self.gelu_layer = nn.GELU()

        self.dropout_layer = nn.Dropout2d()

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

        # x L S4 blocks
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self.batch_norm_layer,
                    self.sep_flex_conv_layer,
                    self.gelu_layer,
                    self.dropout_layer,
                    self.pointwise_linear_layer,
                )
                for _ in range(L)
            ]
        )

    def forward(self, x):
        """
        Standard method of nn.modules we embed also the residual connection
        """
        for block in self.blocks:
            residual = x
            x += residual + block(x)
        return x
