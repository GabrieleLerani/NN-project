import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from hydra import utils
from typing import Optional, Callable, Tuple, Dict, List, cast
from omegaconf import OmegaConf


class PathfinderDataset(torch.utils.data.Dataset):
    """Pathfinder dataset created from a list of images."""

    def __init__(self, transform: Optional[Callable] = None) -> None:
        """
        Args:
            img_list (List[Tuple[str, int]]): List of tuples where each tuple contains
                an image path and its corresponding label.
            transform (Optional[Callable]): Optional transformation function or composition of transformations.
        """
        self.img_list = self.create_imagelist()
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple where the first element is the image tensor
                and the second element is the label.
        """
        img_path, label = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def create_imagelist(self) -> List[Tuple[str, int]]:
        # TODO create image list from directories
        pass


class PathFinderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
        data_type="default",
    ):
        """
        Resolution in [32,64,128,256]
        Level in ["easy", "intermediate", "hard"]
        TODO look at the dir structure

        """
        super().__init__()
        self.data_dir = data_dir
        self.type = cfg.data.dataset
        self.cfg = cfg
        self.num_workers = 0  # for google colab training

        self.data_dir = Path(data_dir)

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 2

    def prepare_data(self):
        # download, not required
        pass

    def setup(self, stage=None):
        self._set_transform()
        self._yaml_parameters()  # TODO set correct params

        self.dataset = PathfinderDataset(self.data_dir, transform=self.transform)
        # compute lengths

        len_dataset = len(self.dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len

        # splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
        )

    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 100)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 1)

        if hidden_channels == 140:
            if self.type == "smnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2976.49)
            elif self.type == "pmnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "train.weight_decay", 0)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2985.63)
        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)

            if self.type == "smnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.01)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2976.49)
            elif self.type == "pmnist":
                OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2985.63)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.data_type == "sequence":
            # If sequential, flatten the input [B, C, Y, X] -> [B, C, -1]
            x, y = batch
            x_shape = x.shape
            # Flatten
            x = x.view(x_shape[0], x_shape[1], -1)
            batch = x, y
        return batch


if __name__ == "__main__":
    # TODO create this list in the dataset init in tghis format
    img_list = [
        # ("path/to/image1.jpg", 0),
        # ("path/to/image2.jpg", 1),
        # Add more image paths and labels
    ]

    # Define any transformations you want to apply
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    dataset = PathfinderDataset(img_list=img_list, transform=transform)
