from torch import nn
from models import SFC
from models import ResS4Net


class CCNN(nn.Module):
    def __init__(self, L, in_channels, out_channels, num_classes):
        """
        Method to init the General Purpose Convolutional Neural Network.
        The model is used to perform spatial data classification.
        The network can be constructed with different block number and channel number.
        """
        super(CCNN, self).__init__()
        # TODO parameters of the model
        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SFC(
            in_channels=in_channels, out_channels=out_channels
        )

        # number of features = the number of output channels of the separable flexible convolutional layer
        self.batch_norm_layer_1 = nn.BatchNorm2d(num_features=out_channels)
        self.gelu_layer = nn.GELU()

        # residual network, modification to the FlexNet architecture
        self.res_s4_net = ResS4Net(
            L=L, in_channels=in_channels, out_channels=out_channels
        )

        self.batch_norm_layer_2 = nn.BatchNorm2d(num_features=out_channels)

        # global average pooling layer
        self.global_avg_pool_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

        # TODO verify if needed because in the paper is written that they perform image classification
        self.linear = nn.Linear(in_features=out_channels, out_features=num_classes)

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        x = self.sep_flex_conv_layer(x)
        x = self.batch_norm_layer_1(x)
        x = self.gelu_layer(x)
        x = self.res_s4_net(x)
        x = self.batch_norm_layer_2(x)
        x = self.global_avg_pool_layer(x)
        x = self.pointwise_linear_layer(x)
        # TODO used to flatten to (N,num_classes) maybe is needed the linear
        x = x.view(x.size(0), -1)

        return x
