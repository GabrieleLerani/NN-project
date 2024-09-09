import pytorch_lightning as L
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Tuple
import numpy as np

class MnistDataModule(L.LightningDataModule):
    def __init__(self, cfg, data_dir: str = "datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.type = cfg.data.dataset
        self.cfg = cfg
        self.num_workers = 0  # for google colab training
        self._yaml_parameters()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def _set_transform(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                transforms.Lambda(
                    lambda x: x.view(1, -1)
                ),  # flatten the image to 784 pixels
            ]
        )

    
        if self.type == "p_mnist":        
            self.permutation = torch.randperm(784)

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 100)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 1)

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
            generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
        )

        if self.cfg.data.reduced_dataset:
            REDUCED_TRAIN_SIZE = 1000
            REDUCED_VAL_SIZE = 200

            mnist_train, _ = random_split(
                mnist_train_full,
                [REDUCED_TRAIN_SIZE, FULL_TRAIN_SIZE - REDUCED_TRAIN_SIZE],
                generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
            )
            mnist_val, _ = random_split(
                mnist_val_full,
                [REDUCED_VAL_SIZE, FULL_VAL_SIZE - REDUCED_VAL_SIZE],
                generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
            )
        else:
            mnist_train, mnist_val = mnist_train_full, mnist_val_full

        print(f"Training set size: {len(mnist_train)}")
        print(f"Validation set size: {len(mnist_val)}")
        return mnist_train, mnist_val
        
        

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.type == "p_mnist":
            # apply permutation 
            x, y = batch
            if x.device != self.permutation.device: # Check if devices match
                self.permutation = self.permutation.to(x.device) # Move permutation to the same device as x
            x = x[:, :, self.permutation]
            batch = x, y
        return batch


    def show_samples(self, num_samples: int = 5):
        dataset = self.mnist_full
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            image, label = dataset[i]
            # image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(f"Label: {label}")
            axes[i].axis("off")
        plt.show()


if __name__ == "__main__":
    mnist = MnistDataModule("datasets", 32)

    mnist.setup("fit")
    mnist.show_samples(3)
