import pytorch_lightning as pl
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Tuple


class Cifar100DataModule(pl.LightningDataModule):
    def __init__(self, cfg, data_dir : str = "datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.type = cfg.data.dataset
        self.cfg = cfg
        self.num_workers = 0 # for google colab training
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = (0.50707516, 0.48654887, 0.44091784),
                    std = (0.26733429, 0.25643846, 0.27615047)
                )
        ])
        self._yaml_parameters()


    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)     
        

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 3)
        OmegaConf.update(self.cfg, "net.out_channels", 100)
        OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
        OmegaConf.update(self.cfg, "net.data_dim", 2)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
            OmegaConf.update(self.cfg, "train.weight_decay", 0.0001)
            OmegaConf.update(self.cfg, "kernel.omega_0", 3521.55)
            
        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            OmegaConf.update(self.cfg, "kernel.omega_0", 679.14)
            

    def setup(self, stage: str):
        
        self.batch_size = self.cfg.train.batch_size
    
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.cifar100_train, self.cifar100_val = self._get_train_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar100_test = CIFAR100(self.data_dir, train=False, transform=self.transform)
            print(f'Test set size: {len(self.cifar100_test)}')

        if stage == "predict":
            self.cifar100_predict = CIFAR100(self.data_dir, train=False)
            print(f'Prediction set size: {len(self.cifar100_predict)}')


    def _get_train_dataset(self) -> Tuple[Dataset, Dataset]:
        FULL_TRAIN_SIZE = 45000
        FULL_VAL_SIZE = 5000
        self.cifar100_full = CIFAR100(self.data_dir, train=True, transform=self.transform)

        # Split the full dataset into train and validation sets
        train_full, val_full = random_split(
            self.cifar100_full,
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
            self.cifar100_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar100_val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar100_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar100_predict, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

    def show_samples(self, num_samples: int = 5):
        dataset = self.cifar100_full
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            image, label = dataset[i]
            #image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()

if __name__ == "__main__":
    cifar100 = Cifar100DataModule()
    
    cifar100.setup("fit")

    cifar100.show_samples(3)