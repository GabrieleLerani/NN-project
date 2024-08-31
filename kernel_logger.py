import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os

class KernelLogger(pl.Callback):
    def __init__(self, model_name : str):
        super().__init__()
        self.model_name = model_name


    def on_train_epoch_end(self, trainer, pl_module):
        kernel = pl_module.get_kernel()
        if kernel is not None:
            kernel = kernel.detach().cpu().numpy()
            # 1D kernel
            if len(kernel.shape) == 3:
                kernel = kernel.squeeze(0)
            # 2D kernel TODO not sure
            elif len(kernel.shape) == 4:
                kernel = kernel.squeeze(0).squeeze(0)

            # TODO
            # Handle different kernel shapes
            # if kernel.shape == (1, 1, 33):
            #     kernel = kernel.reshape(33, 1)  # Reshape to 2D array for visualization
            # elif len(kernel.shape) == 3 and kernel.shape[0] == 1:

            #kernel = kernel.squeeze(0)  # Remove the first dimension if it's 1
            plt.imshow(kernel, cmap='viridis')
            plt.colorbar()
            plt.title(f'{self.model_name}_epoch_{trainer.current_epoch}')
            os.makedirs(f'kernels_{self.model_name}', exist_ok=True)
            plt.savefig(f'kernels_{self.model_name}/kernel_epoch_{trainer.current_epoch}.png')
            plt.close()