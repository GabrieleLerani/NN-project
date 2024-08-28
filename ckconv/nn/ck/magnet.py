from torch import nn
from .linear import LinearLayer
from .create_coordinates import create_coordinates
import torch
import math
import numpy as np

class MFN(nn.Module):
    def __init__(
        self, data_dim: int, hidden_channels: int, out_channels: int, no_layers: int
    ):
        """
        Initializes an instance of the MFN class.
        Args:
            data_dim (int): The dimension of the input data.
            hidden_channels (int): The number of hidden channels in the linear layers.
            out_channels (int): The number of output channels in the final linear layer.
            no_layers (int): The number of hidden layers in the network.
        Returns:
            None
        """
        super(MFN, self).__init__()

        # hidden layers
        self.linearLayer = nn.ModuleList(
            [
                LinearLayer(
                    dim=data_dim,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(no_layers - 1)
            ]
        )

        # output layer
        self.linearLayer.append(
            LinearLayer(
                dim=data_dim,
                in_channels=hidden_channels,
                out_channels=out_channels,
                bias=True,
            )
        )

        self.reweighted_output_layer = False

    def re_weight_output_layer(self, kernel_positions: torch.Tensor, in_channels: int, data_dim: int):
        """
        Re-weights the last layer of the kernel net by factor = gain / sqrt(in_channels * kernel_size).
        Args:
            gain (float): The gain to re-weight the last layer by.
        Returns:
            None
        """

        if not self.reweighted_output_layer:

            # Re weight the last layer of the kernel net
            
            kernel_size = torch.Tensor([*kernel_positions.shape[data_dim:]]).prod().item() # just a way to get the kernel size 
            # [1, 2, 33, 33] -> [33,33] for data_dim=2
            # prod multiplies all elements in the tensor i.e. 33*33 = 1089
            # item converts the tensor to a python number
            
            # define gain / sqrt(in_channels * kernel_size) by Chang et al. (2020)
            factor = 1.0 / math.sqrt(in_channels * kernel_size)

            # get the last layer and re-weight it
            self.linearLayer[-1].layer.weight.data *= factor 

            # set the flag to True so that the output layer is only re-weighted the first time                                          
            self.reweighted_output_layer = True


    def forward(self, x):
        
        h = self.gabor_filters[0](x)
        for l in range(1, len(self.gabor_filters)):
            h = self.gabor_filters[l](x) * self.linearLayer[l - 1](h)

        last = self.linearLayer[-1](h)

        return last


class MAGNet(MFN):
    def __init__(
        self, data_dim: int, hidden_channels: int, out_channels: int, no_layers: int
    ):
        """
        TODO
        """
        super().__init__(data_dim, hidden_channels, out_channels, no_layers)
        self.gabor_filters = nn.ModuleList(
            [
                AnisotropicGaborLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    current_layer=l,
                )
                for l in range(no_layers)
            ]
        )


class AnisotropicGaborLayer(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        current_layer: int,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        super().__init__()

        """
        TODO self.linear(x) in the sine of the forward function assumes an input size of 2 x Nhid
        while self.linear has data_dim has input size (presumably 2)
        """

        self.data_dim = data_dim

        # linear layer
        self.linear = LinearLayer(
            dim=data_dim,
            in_channels=data_dim,
            out_channels=hidden_channels,
            bias=True,
        )

        gamma_dist = torch.distributions.gamma.Gamma(alpha / (current_layer + 1), beta)

        # generate as gamma_dist as data_dim (gamma_x, gamma_y, ...)
        self.gamma = nn.ParameterList(
            [
                nn.Parameter(gamma_dist.sample((hidden_channels, 1)))
                for _ in range(data_dim)
            ]
        )

        normal_dist = torch.distributions.normal.Normal(0, 1)

        # generate as many mi as data_dim (mi_x, mi_y, ...)
        self.mi = nn.ParameterList(
            [
                nn.Parameter(normal_dist.sample((hidden_channels, 1)))
                for _ in range(data_dim)
            ]
        )

        
        scaling_factor = 25.6

        self.linear.weight = nn.Parameter(torch.randn(hidden_channels,data_dim,*((1,) * data_dim)))
        self.linear.weight.data *= scaling_factor * self.gamma[0].view(
            *self.gamma[0].shape, *((1,) * data_dim)
        )

        self.linear.bias = nn.Parameter(torch.randn(hidden_channels))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        """
        TODO
        """

        # coordinates (x,y,...)
        coord = [x[0][i] for i in range(self.data_dim)]

        # reshaping the parameters
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
            g_envelope *= g_envelopes[i]

        # Squeeze the third dimension
        g_envelope = g_envelope.squeeze(2)

        # computing the sinusoidal
        sinusoidal = torch.sin(self.linear(x))

        return g_envelope * sinusoidal


if __name__ == "__main__":
    # test the model
    data_dim = 2
    model = MAGNet(data_dim=data_dim, hidden_channels=140, out_channels=2, no_layers=3)

    x = create_coordinates(
        3, data_dim
    )
    print(f"grid shape: {x.shape}")
    print(x)
    x = model(x)
    print(f"results : {x}")
    print(x.shape)
 
