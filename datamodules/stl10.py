import pytorch_lightning as pl
from torchvision.datasets import STL10
from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torchvision import transforms
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from typing import Tuple


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, cfg, data_dir : str = "datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        self.num_workers = 0
        self.generator = torch.Generator(device=self.cfg.train.accelerator).manual_seed(42)
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = (0.44671097, 0.4398105, 0.4066468),
                    std = (0.2603405, 0.25657743, 0.27126738)
                )
        ])
        self._yaml_parameters()


    def prepare_data(self):
        # Download
        STL10(self.data_dir, split="train", download=True)
        STL10(self.data_dir, split="test", download=True)


    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 64)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 3)
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 2)
        OmegaConf.update(self.cfg, "kernel.omega_0", 954.28)
        OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            OmegaConf.update(self.cfg, "kernel.kernel_size", 33)

        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
            OmegaConf.update(self.cfg, "kernel.kernel_size", 31)


    def setup(self, stage: str):
        self.batch_size = self.cfg.train.batch_size

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.stl10_train, self.stl10_val = self._get_train_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.stl10_test = STL10(self.data_dir, split="test", transform=self.transform)
            print(f'Test set size: {len(self.stl10_test)}')

        if stage == "predict":
            self.stl10_predict = STL10(self.data_dir, split="test", transform=self.transform)
            print(f'Prediction set size: {len(self.stl10_predict)}')


    def _get_train_dataset(self) -> Tuple[Dataset, Dataset]:

        FULL_TRAIN_SIZE = 4500
        FULL_VAL_SIZE = 500

        self.stl10_full = STL10(self.data_dir, split="train", transform=self.transform)

        # Split the full dataset into train and validation sets
        train_full, val_full = random_split(
            self.stl10_full,
            [FULL_TRAIN_SIZE, FULL_VAL_SIZE],
            generator=self.generator,
        )

        train, val = train_full, val_full

        print(f"Training set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        return train, val


    def train_dataloader(self):
        return DataLoader(self.stl10_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.stl10_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.stl10_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.stl10_predict,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def show_samples(self, num_samples: int = 5):
        dataset = STL10(self.data_dir, split="train", transform=self.transform)
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            image, label = dataset[i]
            image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()
