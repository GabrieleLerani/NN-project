import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CKNN(nn.modules):
    def __init__(self):
        """
        Method to init the Continuous Kernel parameterized as a Continuous Kernel Neural Network
        """
        super().__init__()
        raise NotImplementedError("No implemented yet")

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        raise NotImplementedError("No implemented yet")
