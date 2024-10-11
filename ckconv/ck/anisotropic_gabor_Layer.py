from torch import nn
from .linear import LinearLayer
import torch
import numpy as np

class AnisotropicGaborLayer(nn.Module):
    """
    This class implements an Anisotropic Gabor Layer, a type of layer used in the MAGNets.

    Attributes:
        data_dim (int): The dimension of the input data.
        linear (LinearLayer): A linear layer that applies a linear transformation to the input data.
        gamma (nn.ParameterList): A list of learnable parameters that control the scale of the Gabor filters.
        mi (nn.ParameterList): A list of learnable parameters that control the center of the Gabor filters.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        current_layer: int,
        causal: bool,
        omega_0: float = 2976.49,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        """
        Initializes an instance of the AnisotropicGaborLayer class.

        Args:
            data_dim (int): The dimension of the input data.
            hidden_channels (int): The number of hidden channels in the linear layer.
            current_layer (int): The current layer number, used to adjust the scale of the Gabor filters.
            causal (bool): A flag indicating whether the layer is causal or not.
            omega_0 (float, optional): The base frequency of the Gabor filters. Defaults to 2976.49.
            alpha (float, optional): The shape parameter of the gamma distribution used to sample the scale of the Gabor filters. Defaults to 6.0.
            beta (float, optional): The rate parameter of the gamma distribution used to sample the scale of the Gabor filters. Defaults to 1.0.
        """
        super().__init__()

        self.data_dim = data_dim

        # Linear layer
        self.linear = LinearLayer(
            dim=data_dim,
            in_channels=data_dim,
            out_channels=hidden_channels,
            bias=True,
        )

        gamma_dist = torch.distributions.gamma.Gamma(alpha / (current_layer + 1), beta)

        # Generate as gamma_dist as data_dim (gamma_x, gamma_y, ...)
        self.gamma = nn.ParameterList(
            [
                nn.Parameter(gamma_dist.sample((hidden_channels, 1)))
                for _ in range(data_dim)
            ]
        )
        
        # Generate as many mi as data_dim (mi_x, mi_y, ...)
        self.mi = nn.ParameterList(
            [
                nn.Parameter(torch.rand(hidden_channels, 1))
                for _ in range(data_dim)
            ]
        )

        self.linear.weight = nn.Parameter(torch.randn(hidden_channels,data_dim,*((1,) * data_dim)))
        self.linear.bias = nn.Parameter(torch.randn(hidden_channels))

        self.linear.weight.data *= 2 * np.pi * omega_0 * self.gamma[0].view(
            *self.gamma[0].shape, *((1,) * data_dim)
        )
        self.linear.bias.data.fill_(0.0)


    def forward(self, x):

        # Coordinates (x,y,...)
        coord = [x[0][i] for i in range(self.data_dim)]

        # Reshaping the parameters to [1, 1, 1, W, H] if data_dim = 2
        reshaped_coord = [c.view(1, 1, 1, *c.shape) for c in coord]

        reshaped_gamma = [
            g.view(1, *g.shape, *((1,) * (self.data_dim))) for g in self.gamma
        ]

        reshaped_mi = [m.view(1, *m.shape, *((1,) * (self.data_dim))) for m in self.mi]
        # -> [1, hidden_channels, 1, 1, 1] if data_dim = 2

        g_envelopes = []
        for i in range(self.data_dim):
            g_envelope = torch.exp(
                -0.5 * (reshaped_gamma[i] * (reshaped_coord[i] - reshaped_mi[i])) ** 2
            )  # Shape: [1, hidden_channels, 20, 20]
            g_envelopes.append(g_envelope)

        # Multiply all the envelopes together
        g_envelope = g_envelopes[0]
        for i in range(1, self.data_dim):
            g_envelope = g_envelope * g_envelopes[i]

        # Squeeze the third dimension
        g_envelope = g_envelope.squeeze(2)

        # Compute the sinusoidal
        sinusoidal = torch.sin(self.linear(x))

        return g_envelope * sinusoidal
