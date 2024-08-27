from torchvision.datasets import STL10
from torch.utils.data import random_split, DataLoader
import torch

class STL10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.stl10_test = STL10(self.data_dir, train=False)
        self.stl10_predict = STL10(self.data_dir, train=False)
        stl10_full = STL10(self.data_dir, train=True)
        self.stl10_train, self.stl10_val = random_split(
            stl10_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.stl10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.stl10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.stl10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.stl10_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
