from torch import nn, sqrt, tensor
from ck import GetLinear
import torch


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
                GetLinear(
                    dim=data_dim,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    stride=1,
                    bias=True,
                )
                for _ in range(no_layers)
            ]
        )

        # output layer
        self.linearLayer.append(
            GetLinear(
                dim=data_dim,
                in_channels=hidden_channels,
                out_channels=out_channels,
                stride=1,
                bias=True,
            )
        )

    def forward(self, x):
        """
        TODO
        """
        h = self.gabor_filters[0](x)
        for l in range(1, len(self.gabor_filters)):
            h = self.gabor_filters[l](x) * self.linearLayer[l - 1](h)

        last = self.linearLayer[-1](h)

        return last


class GaborNet(MFN):
    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        """
        TODO
        """
        super().__init__(data_dim, hidden_channels, out_channels, no_layers)
        self.gabor_filters = nn.ModuleList(
            [
                GaborLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    current_layer=l,
                )
                for l in range(no_layers)
            ]
        )


class GaborLayer(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        current_layer: int,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        super().__init__()

        # linear layer
        self.linear = GetLinear(
            dim=data_dim,
            in_channels=data_dim,
            out_channels=hidden_channels,
            stride=1,
            bias=True,
        )

        self.gamma_x = torch.distributions.gamma.Gamma(
            alpha / current_layer, beta
        ).sample(hidden_channels)
        self.gamma_y = torch.distributions.gamma.Gamma(
            alpha / current_layer, beta
        ).sample(hidden_channels)

        # init the gamma_x and gamma_y parameters for the scaling factor for the horizontal and vertical directions
        # self.gammax = nn.Parameter(torch.randn(out_channels, 1))
        # self.gammay = nn.Parameter(torch.randn(out_channels, 1))

        # # init the mu_x and mu_y parameters for the mean of the Gabor Function for the horizontal and vertical directions
        # self.mux = nn.Parameter(torch.randn(out_channels, 1))
        # self.muy = nn.Parameter(torch.randn(out_channels, 1))

        # # init the frequency components W_g for the orientation and frequency of sinusoidal and b for phase offset
        # self.W = nn.Parameter(torch.randn(out_channels, 2))
        # self.b = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        Standard method of nn.modules
        We implement the 2D gabor function in the paper
        We assume the input is x of shape [x,y]
        gamma of shape : [gammax,gammay]
        mu of shape :
        """
        #
        """ x = x[0:]
        y = y[:0]
        shifted_x = x - self.mux
        shifted_y = y - self.muy

        shifted_scaled_x = shifted_x * self.gammax
        shifted_scaled_y = shifted_y * self.gammay

        # we compute [x,y] vector
        xy = torch.cat((x, y), dim=0)

        # computing the gaussian envelope
        g_exp_envelope = torch.exp(
            -0.5 * ((shifted_scaled_x**2) + (shifted_scaled_y**2))
        ) """

        g_envelope = torch.exp(
            -0.5
            * (
                (self.gamma_x * (x[:, 0] - self.mu_x)) ** 2
                + (self.gamma_y * (x[:1] - self.mu_y)) ** 2
            )
        )

        # computing the sinusoidal
        sinusoidal = torch.sin((torch.matmul(self.W, xy) + self.b))

        return g_envelope * sinusoidal
