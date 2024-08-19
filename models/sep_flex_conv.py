import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SFCNN(nn.modules):
    def __init__(self):
        """
        Method to init the modified FlexConv layer, here is used the continuous kernel
        """
        raise NotImplementedError("No implemented yet")

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        raise NotImplementedError("No implemented yet")
