import pytorch_lightning as pl
import matplotlib.pyplot as plt

class KernelLogger(pl.Callback):
    def __init__(self, model_name : str):
        super().__init__()
        self.model_name = model_name


    #def on_train_epoch_end(self, trainer, pl_module):
    def on_after_backward(self, trainer, pl_module):
        kernel = pl_module.get_kernel()
        if kernel is not None:
            kernel = kernel.detach().cpu().numpy()

            # Normalize the kernel to [0, 1]
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
                
            # 2D kernel (1,in_channels, ks, ks)
            if len(kernel.shape) == 4:
                kernel = kernel[0]

            # Add to TensorBoard
            trainer.logger.experiment.add_image("kernel", kernel, global_step=trainer.global_step)