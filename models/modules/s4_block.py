from torch import nn
from models import SFC


class S4Block(nn.Module):
    def __init__(self, L, in_channels, out_channels):
        """
        Method to init the residual network composed of x L S4 block
        """
        super(S4Block, self).__init__()

        self.batch_norm_layer = nn.BatchNorm2d(num_features=in_channels)

        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SFC(
            in_channels=in_channels, out_channels=out_channels
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
