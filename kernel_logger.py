import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

class KernelLogger(pl.Callback):
    def __init__(self, model_name : str):
        super().__init__()
        self.model_name = model_name


    def on_train_epoch_end(self, trainer, pl_module):
        kernel = pl_module.get_kernel()
        if kernel is not None:
            kernel = kernel.detach().cpu().numpy()
            # Reduce the kernel to a 2D tensor
            # 1D kernel
            if len(kernel.shape) == 3:
                kernel = kernel.squeeze(0)
            # 2D kernel TODO not sure
            elif len(kernel.shape) == 4:
                kernel = kernel.squeeze(0).squeeze(0)

            # Normalize the kernel to [0, 1]
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            
            # Convert to RGB format for TensorBoard (if the kernel is 2D)
            if len(kernel.shape) == 2:  # If kernel is 2D
                kernel = np.expand_dims(kernel, axis=0)  # Add channel dimension
                kernel = np.repeat(kernel, 3, axis=0)  # Repeat for 3 color channels

            # Add to TensorBoard
            trainer.logger.experiment.add_image("kernel", kernel, global_step=trainer.global_step)