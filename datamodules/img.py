from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf


class ImageDataModule(pl.LightningDataModule):
    """
    Image Classification Dataset from LRA benchmarks, exploiting Cifar10 dataset (1D or 2D)
    Image (Cifar black & white)
    """

    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
    ):

        super().__init__()

        self.cfg = cfg

        # Save parameters to self
        self.data_dir = Path(data_dir) / "IMG_LRA"
        self.num_workers = 0
        self.generator = torch.Generator(self.cfg.train.accelerator).manual_seed(42)
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_img_lra"
        )

        self.val_split = 0.1

        self.type = self.cfg.data.dataset

        self._yaml_parameters()


    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.49139968, 0.48215841, 0.44653091),
                    std=(0.24703223, 0.24348513, 0.26158784),
                ),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.to(torch.float)),
                transforms.Lambda(lambda x: x / 255.0),
            ]
        )
        if self.type == "s_image":
            # Flatten the image to 1024 pixels
            self.transform.transforms.append(
                transforms.Lambda(lambda x: x.view(1, -1))
            )

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 10)

        if hidden_channels == 140:

            if self.type == "image":
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2085.43)
                OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 33)

            elif self.type == "s_image":
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "train.weight_decay", 0)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 4005.15)
                OmegaConf.update(self.cfg, "kernel.kernel_size", -1)

        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)

            if self.type == "image":
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2306.08)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 33)
            elif self.type == "s_image":
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "kernel.omega_0", 4005.15)
                OmegaConf.update(self.cfg, "kernel.kernel_size", -1)

    def prepare_data(self):
        # Download
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
            self.test_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform)
            print(f'Test set size: {len(self.cifar10_test)}')

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
            generator=self.generator,
        )

        train, val = train_full, val_full

        print(f"Training set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        return train, val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
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