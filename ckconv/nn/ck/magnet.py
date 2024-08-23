from torch import nn
from linear import GetLinear
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
                for _ in range(no_layers - 1)
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
        self.linear = GetLinear(
            dim=data_dim,
            in_channels=data_dim,
            out_channels=hidden_channels,
            stride=1,
            bias=True,
        )

        self.gamma_x = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha / (current_layer + 1), beta).sample(
                (hidden_channels, 1)
            )
        )

        self.gamma_y = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha / (current_layer + 1), beta).sample(
                (hidden_channels, 1)
            )
        )

        # they are initialized according to normal distribution
        normal_dist = torch.distributions.normal.Normal(
            0, 1
        )  # Create a normal distribution with mean 0 and std 1
        self.mi_x = nn.Parameter(normal_dist.sample((hidden_channels, 1)))
        self.mi_y = nn.Parameter(normal_dist.sample((hidden_channels, 1)))

        # init the frequency components W_g for the orientation and frequency of sinusoidal and b for phase offset

        # self.linear.weight.data = nn.Parameter(torch.randn(hidden_channels,data_dim,1,1))
        # self.linear.bias = nn.Parameter(torch.randn(hidden_channels))

        self.linear.weight.data *= self.gamma_x.view(
            *self.gamma_x.shape, *((1,) * data_dim)
        )

        print(f"Size {self.linear.weight.shape}")

        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Standard method of nn.modules
        We implement the 2D gabor function in the paper
        We assume the input is x of shape [batch_size,2]
        """
        # [1,2,20,20]
        # coord = []
        # for i in range(self.data_dim):
        #     coord.append(self.x[0][i])

        x_coord = x[0, 0, :, :]
        y_coord = x[0, 1, :, :]

        # reshaping the parameters
        reshaped_x = x_coord.view(1, 1, *x_coord.shape)
        reshaped_y = y_coord.view(1, 1, *y_coord.shape)

        reshaped_gamma_x = self.gamma_x.view(
            1, *self.gamma_x.shape, *((1,) * (self.data_dim - 1))
        )

        reshaped_gamma_y = self.gamma_y.view(
            1, *self.gamma_y.shape, *((1,) * (self.data_dim - 1))
        )

        reshaped_mi_x = self.mi_x.view(
            1, *self.mi_x.shape, *((1,) * (self.data_dim - 1))
        )

        reshaped_mi_y = self.mi_y.view(
            1, *self.mi_y.shape, *((1,) * (self.data_dim - 1))
        )
        print(f"mix {reshaped_mi_x.shape}")
        g_envelope_x = torch.exp(
            -0.5 * (reshaped_gamma_x * (reshaped_x - reshaped_mi_x)) ** 2
        )  # Shape: [1, hidden_channels, 20, 20]

        g_envelope_y = torch.exp(
            -0.5 * (reshaped_gamma_y * (reshaped_y - reshaped_mi_y)) ** 2
        )  # Shape: [1, hidden_channels, 20, 20]

        g_envelope = g_envelope_x * g_envelope_y

        # TODO error gaussian envelope shape must be [1, hidden_channels, 20, 20] and not [1, hidden_channels,2, 20, 20])

        # reshaped_gamma_x = self.gamma_x.view(1, *self.gamma_x.shape, *((1,) * self.data_dim)) # -> 1, a, b, 1, 1, 1)
        # reshaped_gamma_y = self.gamma_y.view(1, *self.gamma_y.shape, *((1,) * self.data_dim))

        # reshaped_mi_x = self.mi_x.view(1, *self.mi_x.shape, *((1,) * self.data_dim))
        # reshaped_mi_y = self.mi_y.view(1, *self.mi_y.shape, *((1,) * self.data_dim))

        # computing the gaussian envelope
        # g_envelope = torch.exp(
        #     -0.5
        #     * (
        #         (reshaped_gamma_x * (x.unsqueeze(1) - reshaped_mi_x)) ** 2
        #         + (reshaped_gamma_y * (x.unsqueeze(1) - reshaped_mi_y)) ** 2
        #     )
        # )

        # computing the sinusoidal
        sinusoidal = torch.sin(self.linear(x))

        print(f"g_envelope: {g_envelope.shape}")
        print(f"sinusoidal shape: {x.shape}")

        return g_envelope * sinusoidal


def linspace_grid(grid_sizes):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    tensors = []
    for size in grid_sizes:
        tensors.append(torch.linspace(-1, 1, steps=size))

    grid = torch.stack(torch.meshgrid(*tensors), dim=0)

    return grid


if __name__ == "__main__":
    # test the model
    data_dim = 2
    model = MAGNet(data_dim=data_dim, hidden_channels=32, out_channels=50, no_layers=3)

    x = linspace_grid(
        [
            20,
        ]
        * data_dim
    ).unsqueeze(0)
    print(f"grid shape: {x.shape}")

    x = model(x)
    print(x.shape)
