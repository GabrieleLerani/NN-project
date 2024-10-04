from torch import nn
from .linear import LinearLayer
from .create_coordinates import create_coordinates
from .anisotropic_gabor_Layer import AnisotropicGaborLayer
import torch
import math


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
                for _ in range(no_layers)
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
    """
    This class is a subclass of the MFN (Multiplicative Filter Network) class.
    It is used to create a Magnetic Network (MAGNet) with anisotropic Gabor filters.
    """
    def __init__(
        self, data_dim: int, hidden_channels: int, out_channels: int, no_layers: int, omega_0: float, causal: bool
    ):
        """
        Initializes an instance of the MAGNet class.
        Args:
            data_dim (int): The dimension of the input data.
            hidden_channels (int): The number of hidden channels in the linear layers.
            out_channels (int): The number of output channels in the final linear layer.
            no_layers (int): The number of hidden layers in the network.
            omega_0 (float): The base frequency of the Gabor filters.
        Returns:
            None
        """
        super().__init__(data_dim, hidden_channels, out_channels, no_layers)
        self.gabor_filters = nn.ModuleList(
            [
                AnisotropicGaborLayer(
                    data_dim=data_dim,
                    hidden_channels=hidden_channels,
                    causal=causal,
                    current_layer=l,
                    omega_0=omega_0,
                )
                for l in range(no_layers + 1)
            ]
        )


# if __name__ == "__main__":
#     # test the model
#     data_dim = 2
#     model = MAGNet(data_dim=data_dim, hidden_channels=140, out_channels=2, no_layers=3)

#     x = create_coordinates(
#         3, data_dim
#     )
#     print(f"grid shape: {x.shape}")
#     print(x)
#     x = model(x)
#     print(f"results : {x}")
#     print(x.shape)
 
