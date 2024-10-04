import torch

def create_coordinates(kernel_size, data_dim):
    """
    This function creates a grid of coordinates for a given kernel size and data dimension.
    The grid is created by linearly spacing values from -1 to 1 and then reshaping and broadcasting them to match the kernel size.

    Args:
        kernel_size (int): The size of the kernel.
        data_dim (int): The dimensionality of the data.

    Returns:
        torch.Tensor: A grid of coordinates with shape [1, data_dim, kernel_size, ..., kernel_size].

    Example:
        >>> create_coordinates(3, 2) # tensor of shape [1, 2, 3, 3]
        tensor([[[[-1., -1., -1.],
                  [ 0.,  0.,  0.],
                  [ 1.,  1.,  1.]],

                  [[-1.,  0.,  1.],
                  [-1.,  0.,  1.],
                  [-1.,  0.,  1.]]]])
    """
    
    values = torch.linspace(-1, 1, steps=kernel_size)   # i.e tensor([-1, ... ,1]) 
    positions = [values for _ in range(data_dim)]      # i.e [tensor([-1, ... ,1])], tensor([-1, ... ,1])]

    grids = []
    for i, t in enumerate(positions):
        shape = [1] * data_dim      # i.e [1, 1] for data_dim = 2
        shape[i] = -1               # shape = [-1, 1] i = 0
                                    # shape = [1, -1] i = 1
        
        t_reshaped = t.view(*shape) # t_reshaped_0 [3, 1], t_reshaped_1 [1, 3]
        
        t_broadcasted = t_reshaped.expand(* [kernel_size] * data_dim) # expand dimension to match [3,3]
        grids.append(t_broadcasted)

    grids = torch.stack(grids, dim=0).unsqueeze(0)  # stack along a new dimension [2,3,3] and another dimension [1,2,3,3]
    
    return grids