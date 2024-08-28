from torch import nn, optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics

from models.modules import S4Block
from ckconv.nn import SepFlexConv
from ckconv.nn.ck import GetLinear
from models.modules.utils import GetBatchNormalization
from models.modules.utils import GetAdaptiveAvgPool
from omegaconf import OmegaConf

class CCNN(pl.LightningModule):
    """
    CCNN architecture (Romero et al., 2022) as defined in the original paper.

          input
            |
        SepFlexConv
            |
        BatchNorm
            |
           GELU
            |
        L x S4Block
            |
        BatchNorm
            |
        GlobalAvgPool
            |
        PointwiseLinear
            |
          output
    """
    def __init__(
        self,  
        in_channels: int,
        out_channels: int, 
        data_dim: int, 
        cfg: OmegaConf
    ):
        super(CCNN, self).__init__()

        self.no_blocks = cfg.net.no_blocks
        hidden_channels = cfg.net.hidden_channels

        self.learning_rate = cfg.train.learning_rate
        self.warmup_epochs = cfg.train.warmup_epochs
        self.epochs = cfg.train.epochs
        self.start_factor = cfg.train.start_factor
        self.end_factor = cfg.train.end_factor

        # separable flexible convolutional layer
        self.sep_flex_conv_layer = SepFlexConv(
            data_dim=data_dim,
            in_channels=in_channels, 
            net_cfg=cfg.net,
            kernel_cfg=cfg.kernel
        )
        # batch normalization layer
        self.batch_norm_layer_1 = GetBatchNormalization(data_dim=data_dim, num_features=hidden_channels)
        # gelu layer
        self.gelu_layer = nn.GELU()
        # s4blocks
        self.blocks = []
        for _ in range(self.no_blocks):
            s4 = S4Block(in_channels=hidden_channels, out_channels=hidden_channels, data_dim=data_dim, net_cfg=cfg.net, kernel_cfg=cfg.kernel)
            self.blocks.append(s4)
        # batch normalization layer
        self.batch_norm_layer_2 = GetBatchNormalization(data_dim=data_dim, num_features=hidden_channels)
        # global average pooling layer (the information of each channel is compressed into a single value)
        self.global_avg_pool_layer = GetAdaptiveAvgPool(data_dim=data_dim, output_size=(1,) * data_dim)
        # pointwise linear convolutional layer
        self.pointwise_linear_layer = GetLinear(data_dim, hidden_channels, out_channels)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        out = self.gelu_layer(
            self.batch_norm_layer_1(
                self.sep_flex_conv_layer(x)
            )
        )

        for i in range(self.no_blocks):
            out = self.blocks[i](out)
        
        out = self.pointwise_linear_layer(self.global_avg_pool_layer(self.batch_norm_layer_2(out)))

        return out.squeeze()

    # Probably works only for sMNIST
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss, scores, y
    
    def configure_optimizers(self):
        # Define the optimizer (AdamW)
        optimizer = optim.adamw(self.parameters(), lr=self.learning_rate)

        # Define the linear learning rate warm-up for 10 epochs
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.warmup_epochs)

        # Define the cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(self.epochs - self.warmup_epochs))

        # Combine the warm-up and cosine annealing using SequentialLR
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_warmup, cosine_scheduler], milestones=[self.warmup_epochs])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}