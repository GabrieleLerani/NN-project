import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Tuple


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, cfg, data_dir : str = "datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.type = cfg.data.dataset
        self.cfg = cfg
        self.num_workers = 0
        self._yaml_parameters()
        self.generator = torch.Generator(device=self.cfg.train.accelerator).manual_seed(42)

    def prepare_data(self):
        # Download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)


    def _set_transform(self):

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = (0.49139968, 0.48215841, 0.44653091),
                    std = (0.24703223, 0.24348513, 0.26158784)
                )
        ])

        if self.type == "s_cifar10":
            # Flatten the image to 1024 pixels, preserve 3 channels
            self.transform.transforms.append(transforms.Lambda(lambda x: x.view(3, -1)))


    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 3)
        OmegaConf.update(self.cfg, "net.out_channels", 10)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.learning_rate", 0.02)

            if self.type == "cifar10":
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "train.weight_decay", 0.0001)
                OmegaConf.update(self.cfg, "kernel.omega_0", 1435.77)
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 33)
            elif self.type == "s_cifar10":
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.0)
                OmegaConf.update(self.cfg, "train.weight_decay", 0)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2386.49)
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "kernel.kernel_size", -1)

        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)

            if self.type == "cifar10":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.15)
                OmegaConf.update(self.cfg, "kernel.omega_0", 1435.77)
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 31)
            elif self.type == "s_cifar10":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.25)
                OmegaConf.update(self.cfg, "kernel.omega_0", 4005.15)
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "kernel.kernel_size", -1)


    def setup(self, stage: str):
        self._set_transform()

        self.batch_size = self.cfg.train.batch_size

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.cifar10_train, self.cifar10_val = self._get_train_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            print(f'Test set size: {len(self.cifar10_test)}')

        if stage == "predict":
            self.cifar10_predict = CIFAR10(self.data_dir, train=False)
            print(f'Prediction set size: {len(self.cifar10_predict)}')

    def _get_train_dataset(self) -> Tuple[Dataset, Dataset]:
        FULL_TRAIN_SIZE = 45000
        FULL_VAL_SIZE = 5000
        self.cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)

        # Split the full dataset into train and validation sets
        train_full, val_full = random_split(
            self.cifar10_full,
            [FULL_TRAIN_SIZE, FULL_VAL_SIZE],
            generator=torch.Generator(self.cfg.train.accelerator).manual_seed(42),
        )

        train, val = train_full, val_full

        print(f"Training set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        return train, val

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            generator=self.generator
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            generator=self.generator
        )

    def test_dataloader(self):
        return DataLoader(self.cifar10_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def show_samples(self, num_samples: int = 5):
        dataset = self.cifar10_full
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            image, label = dataset[i]
            #image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()