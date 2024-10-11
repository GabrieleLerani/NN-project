import pytorch_lightning as pl
import torch


class KernelLogger(pl.Callback):
    def __init__(self, model_name : str):
        super().__init__()
        self.model_name = model_name


    def on_train_epoch_end(self, trainer, pl_module):

        kernel = pl_module.get_kernel()
        log_kernel = pl_module.get_log_kernel()
        log_mask = pl_module.get_log_mask()

        if kernel is None or log_kernel is None or log_mask is None:
            return

        kernel = kernel.detach().cpu().numpy()
        log_kernel = log_kernel.detach().cpu().numpy()
        log_mask = log_mask.detach().cpu().numpy()

        max_all = max(kernel.max(), log_kernel.max(), log_mask.max())
        min_all = max(kernel.min(), log_kernel.min(), log_mask.min())


        kernel = (kernel - min_all) / (max_all - min_all)
        kernel = torch.round(torch.tensor(kernel * 255))

        # 2D kernel (1,in_channels, ks, ks)
        if len(kernel.shape) == 4:
            kernel = kernel[0]

        # Add to TensorBoard
        trainer.logger.experiment.add_image("kernel", kernel, global_step=trainer.global_step)


        log_kernel = (log_kernel - min_all) / (max_all - min_all)
        log_kernel = torch.round(torch.tensor(log_kernel * 255))

        # 2D log_kernel (1,in_channels, ks, ks)
        if len(log_kernel.shape) == 4:
            log_kernel = log_kernel[0]

        # Add to TensorBoard
        trainer.logger.experiment.add_image("log_kernel", log_kernel, global_step=trainer.global_step)


        log_mask = (log_mask - min_all) / (max_all - min_all)
        log_mask = torch.round(torch.tensor(log_mask * 255))

        # 2D log_mask (1,in_channels, ks, ks)
        if len(log_mask.shape) == 4:
            log_mask = log_mask[0]

        # Add to TensorBoard
        trainer.logger.experiment.add_image("log_mask", log_mask, global_step=trainer.global_step)