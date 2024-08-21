from torch import nn, sqrt, tensor
import torch.nn.functional as f
from models import AGL


class KGN(nn.Module):
    def __init__(
        self, in_channels, out_channels, out_hidden, gain, kernel_size, N, num_layers=3
    ):
        """
        Method to init the Continuous Kernel parameterized as a Continuous Kernel Neural Network.
        GKernel. Our kernel generator network is parameterized as a 3-layer MAGNet
        with 32 hidden units for the CCNN4,140 models, and 64 hidden units for the larger CCNN6,380 models.
        """
        super(KGN, self).__init__()

        self.scaling_factor = gain / sqrt(tensor(in_channels * kernel_size))
        self.N = N
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.layers = nn.ModuleList()

        self.layers.append(AGL(out_hidden))

        for l in range(1, num_layers):
            self.layers.append(nn.Linear(out_hidden, out_channels))

        self.final_layer = nn.Linear(out_hidden, out_channels) * gain

    def forward(self, x):
        """
        Standard method of nn.modules
        """

        # normalize coordinates
        x = self.normalize_coordinates(x, self.N)
        x = self.layers[0](x)

        for l in range(1, self.num_layers):
            x = f.relu(self.layers[l](x))

        x = self.reweight_final_layer(self.final_layer(x))
        return x

    def reweight_final_layer(self, x):
        """
        Method to reweight the last layer to avoid problems like vanishing or exploding gradients
        """
        return x * self.scaling_factor

    def normalize_coordinates(self, p, N):
        """
        Method to normalize coordinates given to the GKernel network in input to speed up
        """
        return 2 * (p / N) - 1
