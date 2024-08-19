import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SFCNN, S4


class CCNN(nn.modules):
    def __init__(self, blocks, in_channels, out_channels):
        """
        Method to init the General Purpose Convolutional Neural Network.
        The model is used to perform spatial data classification.
        We construct the CNN_4,110 or the CNN_6,380
        """
        self.layers = []
        self.s4_blocks = []

        # separated flexible convolutional layer
        self.sep_flex_conv_layer = SFCNN()

        self.batch_norm_layer = nn.BatchNorm2d()
        self.gelu_layer = nn.GELU()

        # x L S4 blocks
        for _ in range(blocks):
            self.s4_blocks.append(
                S4(in_channels=in_channels, out_channels=out_channels)
            )

        # global average pooling layer
        self.global_avg_pool_layer = nn.AdaptiveAvgPool2d()

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

        self.layers.append(self.sep_flex_conv_layer)
        self.layers.append(self.batch_norm_layer)
        self.layers.append(self.gelu_layer)
        self.layers.append(self.global_avg_pool_layer)
        self.layers.append(self.pointwise_linear_layer)

        #
        raise NotImplementedError("No implemented yet")

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        raise NotImplementedError("No implemented yet")
