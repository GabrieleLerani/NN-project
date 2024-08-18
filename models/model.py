import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NN(nn.modules):
    def __init__(self):
        """
        Method to init the NN
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        Standard method of nn.modules
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
