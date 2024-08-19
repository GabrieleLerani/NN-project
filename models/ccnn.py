import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CCNN(nn.modules):
    def __init__(self):
        """
        Method to init the General Purpose Convolutional Neural Network.
        The model is used to perform spatial data classification
        """
        raise NotImplementedError("No implemented yet")

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        raise NotImplementedError("No implemented yet")
