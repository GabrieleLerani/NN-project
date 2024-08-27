from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
import torch

class Cifar10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.cifar10_test = CIFAR10(self.data_dir, train=False)
        self.cifar10_predict = CIFAR10(self.data_dir, train=False)
        cifar10_full = CIFAR10(self.data_dir, train=True)
        self.cifar10_train, self.cifar10_val = random_split(
            cifar10_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
