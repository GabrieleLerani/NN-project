import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import MAGNET


class CKNN(nn.modules):
    def __init__(self):
        """
        Method to init the Continuous Kernel parameterized as a Continuous Kernel Neural Network.
        GKernel. Our kernel generator network is parameterized as a 3-layer MAGNet
        with 32 hidden units for the CCNN4,140 models, and 64 hidden units for the larger CCNN6,380 models.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Sigmoid(),
        )
        self.magnets_layer = MAGNET()  # x 3 magnets layer
        raise NotImplementedError("No implemented yet")

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        raise NotImplementedError("No implemented yet")
