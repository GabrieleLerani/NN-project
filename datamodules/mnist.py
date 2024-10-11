import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Tuple
import numpy as np


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, cfg:OmegaConf, data_dir: str = "datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.type = cfg.data.dataset
        self.cfg = cfg
        self.num_workers = 0
        self._yaml_parameters()
        self.generator = torch.Generator(device=self.cfg.train.accelerator).manual_seed(42)

    def prepare_data(self):
        # Download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)


    def _generate_permutation(self):
        np.random.seed(42)
        permutation = np.random.permutation(784)
        return permutation

    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                # Flatten the image to 784 pixels
                transforms.Lambda(
                    lambda x: x.view(1, -1)
                ),
            ]
        )

        if self.type == "p_mnist":
            self.permutation = self._generate_permutation()  # Generate a fixed permutation
            print(self.permutation)
            self.transform.transforms.append(transforms.Lambda(
                        lambda x: x[:,self.permutation]  # Fixed permutation
                    ))

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 100)
        OmegaConf.update(self.cfg, "train.epochs", 136)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 1)
        OmegaConf.update(self.cfg, "kernel.kernel_size", -1)

        if hidden_channels == 140:
            if self.type == "s_mnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2976.49)
            elif self.type == "p_mnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "train.weight_decay", 0)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2985.63)
        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)

            if self.type == "s_mnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2976.49)
            elif self.type == "p_mnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2985.63)

    def setup(self, stage: str):
        self._set_transform()

        self.batch_size = self.cfg.train.batch_size

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            self.mnist_train, self.mnist_val = self._get_train_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
            print(f"Test set size: {len(self.mnist_test)}")

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
            print(f"Prediction set size: {len(self.mnist_predict)}")

    def _get_train_dataset(self) -> Tuple[Dataset, Dataset]:
        FULL_TRAIN_SIZE = 55000
        FULL_VAL_SIZE = 5000
        self.mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

        # Split the full dataset into train and validation sets
        mnist_train_full, mnist_val_full = random_split(
            self.mnist_full,
            [FULL_TRAIN_SIZE, FULL_VAL_SIZE],
            generator=self.generator,
        )
            
        mnist_train, mnist_val = mnist_train_full, mnist_val_full

        print(f"Training set size: {len(mnist_train)}")
        print(f"Validation set size: {len(mnist_val)}")
        return mnist_train, mnist_val


    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            generator=self.generator
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def show_samples(self, num_samples: int = 5):
        dataset = self.mnist_full
        for i in range(num_samples):
            image, label = dataset[i]
            # Create the plot
            plt.figure(figsize=(10,10))  # Wide plot to fit 784 pixels
            plt.plot(image, marker='o', markersize=20, linestyle='-',)

            # Add title and labels
            plt.title(f"Flattened Image of Label: {label}")

        plt.show()