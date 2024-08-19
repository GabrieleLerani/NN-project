import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SFCNN


class S4(nn.modules):
    def __init__(self, in_channels, out_channels):
        """
        Method to init the S4 block
        """
        super().__init__()

        self.batch_norm_layer = nn.BatchNorm2d()

        # separated flexible convolutional layer
        self.sep_flex_conv_layer = SFCNN()

        self.gelu_layer = nn.GELU()

        self.dropout_layer = nn.Dropout2d()

        # pointwise linear convolutional layer
        self.pointwise_linear_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

        raise NotImplementedError("No implemented yet")

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        raise NotImplementedError("No implemented yet")
