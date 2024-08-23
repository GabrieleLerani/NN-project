from torch import nn
from ck import MAGNet
from ck import linspace_grid
import torch


class CKConv(nn.Module):
    def __init__(
        self, 
        data_dim: int, 
        hidden_channels: int, 
        out_channels: int, 
        no_layers: int,
        kernel_size: int

    ) :
        """
        TODO
        """
        super().__init__()

        self.KernelNet = MAGNet(
                data_dim=data_dim,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                no_layers=no_layers,
            )
        
        self.kernel_size = kernel_size
        self.kernel_positions = torch.zeros(1)
        

    
    def GetRelPositions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if (
            self.kernel_positions.shape[-1] == 1 # Only for the first time
        ):  # The conv. receives input signals of length > 1

                
            # Creates the vector of relative positions
            kernel_positions = linspace_grid(
                grid_sizes=torch.Tensor[self.kernel_size].repeat(self.data_dim)
            )
            # -> Grid sized: [kernel_size * data_dim]
            # -> kernel_positions : [dim, kernel_size, kernel_size]  
            

            kernel_positions = kernel_positions.unsqueeze(0)
            # -> kernel_positions sized: [1, dim, kernel_size, kernel_size]

            self.kernel_positions = kernel_positions.type_as(self.kernel_positions)
            # -> With form: [batch_size=1, dim, x_dimension, y_dimension, ...]

            # Save the step size for the calculation of dynamic cropping
            # The step is max - min / (no_steps - 1)
            self.linspace_stepsize = (
                (1.0 - (-1.0)) / (self.train_length[0] - 1)
            ).type_as(self.linspace_stepsize)
        return self.kernel_positions

    

    def gaussian_mask(
        self,
        kernel_pos: torch.Tensor,
        mask_mean_param: torch.Tensor,
        mask_width_param: torch.Tensor,
        **kwargs,
    ):
        # mask.shape = [1, 1, Y, X] in 2D or [1, 1, X] in 1D
        return torch.exp(
            -0.5
            * (1.0 / (mask_width_param ** 2 + 1e-8) * (kernel_pos - mask_mean_param) ** 2).sum(
                1, keepdim=True
            )
        )



if __name__ == "__main__":
    # test the model
    data_dim = 2
    model = CKConv(data_dim, 3, 1, 2)


