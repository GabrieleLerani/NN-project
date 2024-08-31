import pytorch_lightning as pl
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class SimpleCNN(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # Dummy input to calculate the correct size
        self.dummy_input = torch.zeros(1, 1, 28, 28)
        self._calculate_output_size()
        self.fc1 = nn.Linear(self.output_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _calculate_output_size(self):
        with torch.no_grad():
            x = self.dummy_input
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), -1)
            self.output_size = x.size(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


from pytorch_lightning.callbacks import Callback
import torchvision.utils as vutils


class FeatureMapLogger(Callback):
    def on_batch_end(self, trainer, pl_module):
        batch = next(iter(trainer.datamodule.train_dataloader()))
        x, _ = batch
        x = x.to(pl_module.device)
        features = pl_module.conv1(x)
        grid = vutils.make_grid(features.cpu(), nrow=8, normalize=True, scale_each=True)
        print(f"GRID : {grid}")
        trainer.logger.experiment.add_image(
            "conv1_feature_maps", grid, global_step=trainer.global_step
        )


from pytorch_lightning import LightningDataModule


class MNISTDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage=None):
        self.train_dataset = MNIST(
            root=".", train=True, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)


# Instantiate data module
data_module = MNISTDataModule()
data_module.setup()
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# Initialize TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="simple_cnn")
# Initialize model and callbacks
model = SimpleCNN()
feature_map_logger = FeatureMapLogger()

# Initialize trainer
trainer = Trainer(
    logger=logger, callbacks=[feature_map_logger], max_epochs=1
)  # Adjust epochs as needed
trainer.fit(model, datamodule=data_module)
