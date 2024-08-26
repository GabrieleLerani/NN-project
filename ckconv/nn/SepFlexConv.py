import torch
from torch import nn
from ck import MAGNet
from ck import linspace_grid
from ck import GetLinear
from conv import fftconv
from conv import conv as simple_conv



class SepFlexConv(nn.Module):
    """
    SeparableFlexConv (SepFlexConv) is a depthwise separable version of FlexConv (Romero et al., 2022a)
    
    ConstructMaskedKernel is a continuous version of the kernel whichs multiplied by a Gaussian mask.
    
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
        hidden_channels: int,
        kernel_no_layers: int,
        kernel_hidden_channels: int,
        kernel_size: int = 33,
        conv_type: str = "conv",
        fft_thresold: int = 50,
        bias: bool = False, 
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
            bias (bool, optional): Whether to include bias in the pointwise convolution layer. Defaults to False.
        """
        super().__init__()

        self.kernel_size = kernel_size

        # init relative positions of the kernel
        self.kernel_positions = torch.zeros(1)

        # init random bias with in_channels dimensions
        self.bias = torch.randn(in_channels)
        self.bias.data.fill_(0.0)

        # store variables
        self.data_dim = data_dim
        self.in_channels = in_channels

        # init gaussian mask parameter
        self.mask_mean = torch.nn.Parameter(torch.zeros(data_dim)) # mi = 0
        self.mask_sigma = torch.nn.Parameter(torch.ones(data_dim))

        self.conv_type = conv_type
        self.fft_thresold = fft_thresold

        # Define the kernel net, in our case always a MAGNet
        self.KernelNet = MAGNet(
            data_dim=data_dim,
            hidden_channels=kernel_hidden_channels,
            out_channels=in_channels,
            no_layers=kernel_no_layers,
        )
        
        # Define the pointwise convolution layer (page 4 original paper)
        self.pointwise_conv = GetLinear(
            dim=data_dim,
            in_channels=in_channels,
            out_channels=hidden_channels,
            bias=bias,
        )
        

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
        self.KernelNet.re_weight_output_layer(kernel_positions, self.in_channels, self.data_dim)

        # 3. Get the kernel
        conv_kernel = self.KernelNet(kernel_positions) 

        # 4. Get the mask gaussian mask
        mask = self.gaussian_mask(
            kernel_pos=kernel_positions,
            mask_mean=self.mask_mean,  
            mask_sigma=self.mask_sigma,
        )

        return conv_kernel * mask

    def get_rel_positions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if (
            self.kernel_positions.shape[-1] == 1  # Only for the first time
        ):  # The conv. receives input signals of length > 1

            # Creates the vector of relative positions
            
            kernel_positions = linspace_grid(
                grid_sizes=torch.Tensor([self.kernel_size]).repeat(self.data_dim)
            )
            # -> Grid sized: [kernel_size * data_dim]
            # -> kernel_positions : [dim, kernel_size, kernel_size]

            kernel_positions = kernel_positions.unsqueeze(0)
            # -> kernel_positions sized: [1, dim, kernel_size, kernel_size]

            self.kernel_positions = kernel_positions.type_as(self.kernel_positions)
            # -> With form: [batch_size=1, dim, x_dimension, y_dimension, ...]

            # Save the step size for the calculation of dynamic cropping
            # The step is max - min / (no_steps - 1)
            # TODO : Check cropping
            # self.linspace_stepsize = (
            #     (1.0 - (-1.0)) / (self.train_length[0] - 1)
            # ).type_as(self.linspace_stepsize)
        return self.kernel_positions

    def gaussian_mask(
            self,
            kernel_pos: torch.Tensor,
            mask_mean: torch.Tensor,
            mask_sigma: torch.Tensor
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
        
        # reshape the mask_mean and mask_sigma so that they can be broadcasted
        mask_mean = mask_mean.view(1, self.data_dim, *(1,) * self.data_dim)
        mask_sigma = mask_sigma.view(1, self.data_dim, *(1,) * self.data_dim)
        
        return torch.exp(
            -0.5
            * (
                1.0 / (mask_sigma**2) * (kernel_pos - mask_mean) ** 2
            ).sum(1, keepdim=True)
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
        
        masked_kernel = self.construct_masked_kernel(x)
        print("x shape: ", x.shape)
        print("Conv kernel shape: ", masked_kernel.shape)
        

        size = torch.tensor(masked_kernel.shape[2:]) # -> [33,33] for data_dim=2
        # fftconv is used when the size of the kernel is large enough
        if self.conv_type == "fftconv": #and torch.all(size > self.fft_thresold):
            out = fftconv(x=x, kernel=masked_kernel, bias=self.bias)
        else:
            out = simple_conv(x=x, kernel=masked_kernel, bias=self.bias)

        print("Spatial conv shape: ", out.shape)
        # pointwise convolution where out is the spatial convolution
        out = self.pointwise_conv(out)
    
        return out


def test_1D_sep_flex_conv():
    # Define the parameters
    data_dim = 1
    in_channels = 1
    
    hidden_channels = 140
    kernel_no_layers = 3
    kernel_hidden_channels = 32
    kernel_size = 65
    conv_type = "conv"
    fft_thresold = 50
    bias = False

    # Instantiate the SepFlexConv model
    model = SepFlexConv(
        data_dim=data_dim,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_no_layers=kernel_no_layers,
        kernel_hidden_channels=kernel_hidden_channels,
        kernel_size=kernel_size,
        conv_type=conv_type,
        fft_thresold=fft_thresold,
        bias=bias
    )

    # Create a sample input tensor
    x = torch.ones(size=(100, in_channels, 784)).float()
    # Perform a forward pass
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)


def test_2D_sep_flex_conv():
    # Define the parameters
    data_dim = 2
    in_channels = 3
    hidden_channels = 140
    kernel_no_layers = 3
    kernel_hidden_channels = 32
    kernel_size = 33
    conv_type = "conv" # TODO check with fftconv
    fft_thresold = 50
    bias = False

    # Instantiate the SepFlexConv model
    model = SepFlexConv(
        data_dim=data_dim,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_no_layers=kernel_no_layers,
        kernel_hidden_channels=kernel_hidden_channels,
        kernel_size=kernel_size,
        conv_type=conv_type,
        fft_thresold=fft_thresold,
        bias=bias
    )

    # Create a sample input tensor
    x = torch.ones(size=(64, in_channels, 32, 32)).float()
    
    # Perform a forward pass
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)


if __name__ == "__main__":
    #test_1D_sep_flex_conv()
    test_2D_sep_flex_conv()