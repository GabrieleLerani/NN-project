import torch
from torch import nn


class AGL(nn.Module):
    def __init__(self, out_channels):
        """
        Method to init the anisotropic gabor layer.
        Here the gabor filter operation takes place.
        """
        super(AGL, self).__init__()

        # init the channels

        # init the gamma_x and gamma_y parameters for the scaling factor for the horizontal and vertical directions
        self.gammax = nn.Parameter(torch.randn(out_channels, 1))
        self.gammay = nn.Parameter(torch.randn(out_channels, 1))

        # init the mu_x and mu_y parameters for the mean of the Gabor Function for the horizontal and vertical directions
        self.mux = nn.Parameter(torch.randn(out_channels, 1))
        self.muy = nn.Parameter(torch.randn(out_channels, 1))

        # init the frequency components W_g for the orientation and frequency of sinusoidal and b for phase offset
        self.W = nn.Parameter(torch.randn(out_channels, 2))
        self.b = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        Standard method of nn.modules
        We implement the 2D gabor function in the paper
        """
        # TODO correct 2D or 1D?
        x = x[0:]
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
        )

        # computing the sinusoidal
        sinusoidal = torch.sin((torch.matmul(self.W, xy) + self.b))

        return g_exp_envelope * sinusoidal
