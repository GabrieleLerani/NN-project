from torch import nn, optim
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification

from models.modules import S4Block
from models.modules import TCNBlock
from ckconv.nn import SepFlexConv
from ckconv.nn.ck import LinearLayer
from models.modules.utils import GetBatchNormalization
from models.modules.utils import GetAdaptiveAvgPool
from omegaconf import OmegaConf

class CCNN(pl.LightningModule):
    """
    CCNN architecture (Romero et al., 2022) as defined in the original paper.

    input --> SepFlexConv --> BatchNorm --> GELU --> L x S4Block --> BatchNorm --> GlobalAvgPool -->PointwiseLinear --> output
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
        self.batch_norm_layer = [
            GetBatchNormalization(data_dim=data_dim, num_features=hidden_channels),
            GetBatchNormalization(data_dim=data_dim, num_features=hidden_channels)
        ]
        # gelu layer
        self.gelu_layer = nn.GELU()
        # blocks can be either S4 or TCN
        self.blocks = []
        for _ in range(self.no_blocks):
            if cfg.net.block_type == "s4":
                s4 = S4Block(
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels, 
                    data_dim=data_dim, 
                    net_cfg=cfg.net, 
                    kernel_cfg=cfg.kernel, 
                    dropout=cfg.train.dropout_rate
                )
                self.blocks.append(s4)
            elif cfg.net.block_type == "tcn":
                tcn = TCNBlock(
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels, 
                    data_dim=data_dim, net_cfg=cfg.net, 
                    kernel_cfg=cfg.kernel, 
                    dropout=cfg.train.dropout_rate
                )
                self.blocks.append(tcn)
        
        
        # global average pooling layer (the information of each channel is compressed into a single value)
        self.global_avg_pool_layer = GetAdaptiveAvgPool(data_dim=data_dim, output_size=(1,) * data_dim)
        # pointwise linear convolutional layer
        self.pointwise_linear_layer = LinearLayer(data_dim, hidden_channels, out_channels)

        # define sequencial modules
        self.seq_modules = nn.Sequential(
            self.sep_flex_conv_layer,
            self.batch_norm_layer[0],
            self.gelu_layer,
            *self.blocks,
            self.batch_norm_layer[1],
            self.global_avg_pool_layer,
            self.pointwise_linear_layer
        )

        # init last layer
        torch.nn.init.kaiming_normal_(self.pointwise_linear_layer.layer.weight)
        self.pointwise_linear_layer.layer.bias.data.fill_(value=0.0)

        # define metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=out_channels)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=out_channels)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=out_channels)

        self.train_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=out_channels)
        self.val_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=out_channels)
        self.test_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        
        out = self.seq_modules(x)

        return out.squeeze()

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.train_accuracy(scores, y)
        self.train_f1_score(scores, y)
        metrics_dict = {
            'train_loss': loss,
            'train_accuracy': self.train_accuracy,
            'train_f1_score': self.train_f1_score
        }
        self.log_dict(dictionary=metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.val_accuracy(scores, y)
        self.val_f1_score(scores, y)
        self.log('val_loss', loss)
        self.log('val_f1_score', self.val_f1_score)
        self.log('accuracy', self.val_accuracy)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.test_accuracy(scores, y)
        self.test_f1_score(scores, y)
        
        metrics_dict = {
            'loss': loss,
            'test_f1_score': self.test_f1_score,
            'accuracy': self.test_accuracy,
        }
        self.log_dict(metrics_dict)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        scores = self.seq_modules(x).squeeze()
        preds = torch.argmax(scores, dim=1)
        return preds

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss, scores, y
    
    def configure_optimizers(self):
        # Define the optimizer (AdamW)
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Define the linear learning rate warm-up for 10 epochs
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.warmup_epochs)

        # Define the cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(self.epochs - self.warmup_epochs))

        # Combine the warm-up and cosine annealing using SequentialLR
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_warmup, cosine_scheduler], milestones=[self.warmup_epochs])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_kernel(self):
        sep_flex_conv_layer = self.seq_modules[0]
        if isinstance(sep_flex_conv_layer, SepFlexConv):
            return sep_flex_conv_layer.masked_kernel
        return None