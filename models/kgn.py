from torch import nn
import torch.nn.functional as f
from models import AGL


class KGN(nn.Module):
    def __init__(self, out_channels, out_hidden, num_layers=3):
        """
        Method to init the Continuous Kernel parameterized as a Continuous Kernel Neural Network.
        GKernel. Our kernel generator network is parameterized as a 3-layer MAGNet
        with 32 hidden units for the CCNN4,140 models, and 64 hidden units for the larger CCNN6,380 models.
        """
        super(KGN, self).__init__()

        self.num_layers = num_layers
        self.out_channels = out_channels
        self.layers = nn.ModuleList()

        self.layers.append(AGL(out_hidden))

        for l in range(1, num_layers):
            self.layers.append(nn.Linear(out_hidden, out_channels))

        self.final_layer = nn.Linear(out_hidden, out_channels)

    def forward(self, x, y):
        """
        Standard method of nn.modules
        """
        x = self.layers[0](x, y)

        for l in range(1, self.num_layers):
            x = f.relu(self.layers[l](x))

        x = self.final_layer(x)
        return x
