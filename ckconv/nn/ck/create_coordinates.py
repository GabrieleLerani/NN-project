import torch

def create_coordinates(kernel_size, data_dim):
    
    values = torch.linspace(-1, 1, steps=kernel_size)   # i.e tensor([-1, 1])  
    positions = [values for _ in range(data_dim)]      # i.e [tensor([-1, 1]), tensor([-1, 1])]

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
    



