from pathlib import Path

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from datasets import load_dataset, DatasetDict

from omegaconf import OmegaConf


class ImageDataModule(pl.LightningDataModule):
    """
    Image Classification Dataset from LRA benchmarks, exploiting Cifar10 dataset (1D or 2D)
    """

    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
        val_split=0.0,
    ):

        super().__init__()

        # Save parameters to self
        self.data_dir = Path(data_dir) / "IMAGE_LRA"
        self.num_workers = 7

        self.val_split = val_split

        # Determine data_type
        self.type = cfg.data.dataset
        self.cfg = cfg

        self._yaml_parameters()

    def prepare_data(self):
        if not self.data_dir.is_dir():
            if self.type == "sequence":  # default
                self.dataset = load_dataset(
                    "allenai/lra_image", "cifar10", cache_dir=self.data_dir
                )
            elif self.data_type == "default":
                self.dataset = load_dataset(
                    "allenai/lra_image", "cifar10_images", cache_dir=self.data_dir
                )

    def setup(self, stage=None):

        self._set_transform()

        self.batch_size = self.cfg.train.batch_size

        self.dataset.set_format(type="torch", columns=["input_ids", "label"])

        self.train_dataset, self.test_dataset = (
            self.dataset["train"],
            self.dataset["test"],
        )

        def collate_batch(batch):
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["label"] for data in batch]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in input_ids],
                batch_first=True,
                padding_value=self.pad_token_id,
            )

            if padded_input_ids.size(1) > self.max_length:
                padded_input_ids = padded_input_ids[
                    :, -self.max_length :
                ]  # truncate to max_length
            else:
                # Pad to max_length on the left (if needed)
                padding_size = self.max_length - padded_input_ids.size(1)
                padded_input_ids = torch.nn.functional.pad(
                    padded_input_ids,
                    (padding_size, 0),  # pad on the left
                    value=self.pad_token_id,
                )

            input_tensor = padded_input_ids.unsqueeze(1).float()
            label_tensor = torch.tensor(labels)

            return input_tensor, label_tensor

        self.collate_fn = collate_batch

    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

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

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_dataloader


if __name__ == "__main__":

    cfg = OmegaConf.load("config/config.yaml")
    dm = ImageDataModule(
        cfg=cfg,
        data_dir="./data/datasets",
    )
    dm.prepare_data()
    dm.setup()
    # Retrieve and print a sample
    train_loader = dm.train_dataloader()

    # Get a batch of data
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")

        # Print the first image and label in the batch
        print(f"First image tensor: {images[0]}")
        print(f"First label: {labels[0]}")

        # Break after first batch for demonstration
        break
