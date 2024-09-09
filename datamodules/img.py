from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split


from datasets import load_dataset, DatasetDict
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as F

from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch
from pathlib import Path
from datasets import load_dataset, DatasetDict
from omegaconf import OmegaConf
import numpy as np
from nltk.tokenize import word_tokenize


class ImageDataModule(pl.LightningDataModule):
    """
    Image Classification Dataset from LRA benchmarks, exploiting Cifar10 dataset (1D or 2D)
    Image (Cifar black & white)
    """

    def __init__(
        self,
        cfg,
        data_dir: str = "data/datasets",
    ):

        super().__init__()

        # Save parameters to self
        self.data_dir = Path(data_dir) / "IMG_LRA"
        self.num_workers = 7
        self.batch_size = cfg.train.batch_size
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_img_lra"
        )

        self.val_split = 0.1

        # Determine data_type
        self.type = cfg.data.type
        self.cfg = cfg

        self._yaml_parameters()

    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.to(torch.float)),
                transforms.Lambda(lambda x: x / 255.0),
            ]
        )
        if self.type == "sequence":
            self.transform.transforms.append(
                transforms.Lambda(lambda x: x.view(-1))
            )  # flatten the image to 1024 pixels

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 2)

        if hidden_channels == 140:

            if self.type == "default":
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)

                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2085.43)
                OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)

            elif self.type == "sequence":
                OmegaConf.update(self.cfg, "train.weight_decay", 0)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)

                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 4005.15)
        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)

            if self.type == "default":
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)

                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2306.08)
            elif self.type == "sequence":
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)

                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "kernel.omega_0", 4005.15)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        self._set_transform()

        self.batch_size = self.cfg.train.batch_size

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = self._get_train_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )
            # print(f'Test set size: {len(self.cifar10_test)}')

        if stage == "predict":
            self.test_dataset = CIFAR10(self.data_dir, train=False)
            print(f"Prediction set size: {len(self.cifar10_predict)}")

    def _get_train_dataset(self):
        FULL_TRAIN_SIZE = 45000
        FULL_VAL_SIZE = 5000
        self.cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)

        # Split the full dataset into train and validation sets
        train_full, val_full = random_split(
            self.cifar10_full,
            [FULL_TRAIN_SIZE, FULL_VAL_SIZE],
            generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
        )

        if self.cfg.data.reduced_dataset:
            REDUCED_TRAIN_SIZE = 500
            REDUCED_VAL_SIZE = 100

            train, _ = random_split(
                train_full,
                [REDUCED_TRAIN_SIZE, FULL_TRAIN_SIZE - REDUCED_TRAIN_SIZE],
                generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
            )
            val, _ = random_split(
                val_full,
                [REDUCED_VAL_SIZE, FULL_VAL_SIZE - REDUCED_VAL_SIZE],
                generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
            )
        else:
            train, val = train_full, val_full

        print(f"Training set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        return train, val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":

    cfg = OmegaConf.load("config/config.yaml")

    dm = ImageDataModule(
        cfg=cfg,
        data_dir="data/datasets",
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
        print(f"First image tensor: {images[0]}")
        print(f"First label: {labels[0]}")
        break
