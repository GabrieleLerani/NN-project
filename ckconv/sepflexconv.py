import torch
from torch import nn
from ckconv.ck import MAGNet, create_coordinates, LinearLayer
from ckconv.conv import fftconv, conv
from omegaconf import OmegaConf

class SepFlexConv(nn.Module):
    """
    SeparableFlexConv (SepFlexConv) is a depthwise separable version of FlexConv (Romero et al., 2022a)

    ConstructMaskedKernel outputs a continuous version of the kernel which is multiplied by a Gaussian mask.

    The gaussian mask has learnable parameters and by learning it the model can learn the size of the convolutional kernel.

    The flow is the following:

        input
            |
            |
            |
            -------------- |
            |              |input.length
            |              |
            |    ConstructMaskedKernel
            |              |
            |              |
            SpatialConvolution
                    |
                    |
            DepthwiseConvolution

    """

    def __init__(
        self,
        data_dim: int,
        in_channels: int,
        net_cfg: OmegaConf,
        kernel_cfg: OmegaConf,
    ):
        """
        Initializes the CKConv module.
        Args:
            data_dim (int): The dimensionality of the input data.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            hidden_channels (int): The number of hidden channels.
            kernel_no_layers (int): The number of layers in the kernel network.
            kernel_hidden_channels (int): The number of hidden channels in the kernel network.
            kernel_size (int, optional): The size of the kernel. Defaults to 33.
            conv_type (str, optional): The type of convolution. Defaults to "conv".
            fft_thresold (int, optional): The threshold for using FFT. Defaults to 50.
            bias (bool, optional): Whether to include bias in the pointwise convolution layer. Defaults to True.
        """
        super().__init__()

        # Sep flex conv parameters
        self.data_dim = data_dim
        self.in_channels = in_channels
        hidden_channels = net_cfg.hidden_channels

        # Kernel parameters
        kernel_no_layers = kernel_cfg.kernel_no_layers
        kernel_hidden_channels = kernel_cfg.kernel_hidden_channels
        self.kernel_size = kernel_cfg.kernel_size
        self.conv_type = kernel_cfg.conv_type
        self.fft_threshold = kernel_cfg.fft_threshold

        # Init relative positions of the kernel
        self.kernel_positions = torch.zeros(1)
        # Init the intervals num between kernels positions
        self.positions_intervals_num = 32
        self.linspace_stepsize = torch.zeros(1)

        if net_cfg.bias:
            # Init random bias with in_channels dimensions
            self.bias = torch.randn(in_channels)
            self.bias.data.fill_(0.0)
        else:
            self.bias = None

        # Causal
        self.causal = net_cfg.causal

        # Init gaussian mask parameter
        if self.causal:
            mask_mean_param = torch.ones(data_dim)
        else:
            mask_mean_param = torch.zeros(data_dim)

        mask_width_param = torch.Tensor([0.075] * data_dim)

        # Do not register mask mean
        self.register_buffer("mask_mean_param", mask_mean_param)
        self.mask_width_param = torch.nn.Parameter(mask_width_param)

        # Define the kernel net
        self.KernelNet = MAGNet(
            data_dim=data_dim,
            hidden_channels=kernel_hidden_channels,
            out_channels=in_channels,  # always in channel because separable
            no_layers=kernel_no_layers,
            omega_0=kernel_cfg.omega_0,
            causal=net_cfg.causal
        )

        # Define the pointwise convolution layer (page 4 original paper)
        self.pointwise_conv = LinearLayer(
            dim=data_dim,
            in_channels=in_channels,
            out_channels=hidden_channels,
            bias=net_cfg.bias,
        )

        # Initialize with kaimi
        torch.nn.init.kaiming_normal_(self.pointwise_conv.layer.weight)
        if self.pointwise_conv.layer.bias is not None:
            torch.nn.init._no_grad_fill_(self.pointwise_conv.layer.bias, 0.0)

    def construct_masked_kernel(self, x):
        """
        Construct the masked kernel by multiplying the result of the kernel net with a
        gaussian mask.

        input.length
        |
        GetRelPositions
        RelPositions
        |
        KernelNet
        ConvKernel
        |
        GaussMask
        |
        MaskedKernel
        """

        # 1. Get the relative positions
        kernel_positions = self.get_rel_positions(x)

        # 2 Re-weight the output layer of the kernel net
        self.KernelNet.re_weight_output_layer(
            kernel_positions, self.in_channels, self.data_dim
        )

        # 3. Get the kernel
        conv_kernel = self.KernelNet(kernel_positions)

        # 4. Get the mask gaussian mask
        mask = self.gaussian_mask(
            kernel_pos=kernel_positions,
            mask_mean=self.mask_mean_param,
            mask_sigma=self.mask_width_param,
        )

        self.log_mask = mask
        self.log_kernel = conv_kernel

        return conv_kernel * mask

    def get_rel_positions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        This method is responsible for creating and managing the kernel positions used in the convolution process.
        It checks if the kernel positions need to be initialized or updated based on the input signal length.

        Args:
            x (torch.Tensor): The input tensor to the SepFlexConv layer.

        Returns:
            torch.Tensor: The tensor of relative positions to be used in the KernelNet.
        """
        if (
            self.kernel_positions.shape[-1] == 1  # Only for the first time
        ):

            if self.kernel_size == -1:
                self.kernel_size = x.shape[-1]

            # Creates the vector of relative positions
            kernel_positions = create_coordinates(
                kernel_size=self.kernel_size,
                data_dim=self.data_dim,
            )

            # -> Grid sized: [kernel_size] * data_dim
            # -> kernel_positions : [1, dim, kernel_size, kernel_size]

            self.kernel_positions = kernel_positions.type_as(self.kernel_positions)
            # -> With form: [batch_size=1, dim, x_dimension, y_dimension, ...]

        return self.kernel_positions

    def gaussian_mask(
        self,
        kernel_pos: torch.Tensor,
        mask_mean: torch.Tensor,
        mask_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates a Gaussian mask based on the given parameters.
        Args:
            kernel_pos (torch.Tensor): The position of the kernel.
            mask_mean (torch.Tensor): The mean value of the mask.
            mask_sigma (torch.Tensor): The standard deviation of the mask.
            Returns:
                torch.Tensor: The generated Gaussian mask of [1, 1, Y, X] in 2D or [1, 1, X] in 1D

        Example 2D:
            if kernel_pos.shape = [1, 2, 33, 33] and mask_mean.shape = [1, 2]
            in order to sum them you need to reshape mask_mean to [1, 2, 1, 1]
            Then you sum over the first dimension and the output will be [1, 1, 33, 33]
        """

        # Reshape the mask_mean and mask_sigma so that they can be broadcasted
        mask_mean = mask_mean.view(1, -1, *(1,) * self.data_dim)
        mask_sigma = mask_sigma.view(1, -1, *(1,) * self.data_dim)

        return torch.exp(
            -0.5
            * (1.0 / (mask_sigma ** 2 + 1e-8) * (kernel_pos - mask_mean) ** 2).sum(
                1, keepdim=True
            )
        )

    def forward(self, x):
        """
        Forward pass of the SepFlexConv model.
        Args:
            x (torch.Tensor): The input tensor.
        Example 2D:
            1. x.shape = [64, 140, 32, 32]
            2. masked_kernel.shape = [1, 140, 33, 33]
            3. spatial convolution between x and masked kernel -> [64, 140, 32, 32]
            4. Pointwise convolution -> [64, 140, 32, 32]
        """

        self.masked_kernel = self.construct_masked_kernel(x)

        size = torch.tensor(self.masked_kernel.shape[2:])
        # fftconv is used when the size of the kernel is large enough
        if self.conv_type == "fftconv" and torch.all(size > self.fft_threshold):
            out = fftconv(x=x, kernel=self.masked_kernel, bias=self.bias)
        else:
            out = conv(
                x=x, kernel=self.masked_kernel, bias=self.bias, causal=self.causal
            )

        # Pointwise convolution
        out = self.pointwise_conv(out)

        return out