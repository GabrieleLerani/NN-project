import pytorch_lightning as L
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, TensorDataset
from .utils import split_data, load_data_from_partition, save_data, normalise_data
import torch
import os
from tqdm import tqdm
from torch.utils.data import random_split

from omegaconf import OmegaConf
from datasets import load_dataset, DatasetDict, Dataset

from pathlib import Path
from torchvision import transforms


class SpeechCommandsDataModule(L.LightningDataModule):

    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
    ):

        super().__init__()

        # Save parameters to self
        self.data_dir = Path(data_dir) / "SPEECH"
        self.num_workers = 7
        self.batch_size = cfg.train.batch_size
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_img_lra"
        )

        self.val_split = 0.1
        self.train_split = 0.9

        # Determine data_type
        self.type = cfg.data.dataset

        assert self.type in ["speech_raw", "speech_mfcc"]
        self.cfg = cfg

        self._yaml_parameters()

    def _set_transform(self):

        if self.type == "speech_raw":
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(
                        lambda x: x.unsqueeze(0)
                    ),  # add channel dimension
                    transforms.Lambda(
                        lambda x: x.to(torch.float32)
                    ),  # convert to float
                    transforms.Normalize(mean=0.0, std=1.0),  # normalize to [0, 1]
                ]
            )
        elif self.type == "speech_mfcc":
            self.transform = transforms.Compose(
                [
                    torchaudio.transforms.MFCC(
                        log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
                    ),
                    transforms.Lambda(
                        lambda x: x.to(torch.float32)
                    ),  # convert to float
                    transforms.Normalize(mean=0.0, std=1.0),  # normalize to [0, 1]
                ]
            )

    def _yaml_parameters(self):
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 1)
        OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
        OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
        OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)

        # 140 and 380 hidden_channels have same parameters
        if self.type == "speech_raw":
            OmegaConf.update(self.cfg, "net.in_channels", 1)
            OmegaConf.update(self.cfg, "train.batch_size", 20)
            OmegaConf.update(self.cfg, "train.epochs", 160)
            OmegaConf.update(self.cfg, "kernel.omega_0", 1295.61)
        elif self.type == "speech_mfcc":
            OmegaConf.update(self.cfg, "net.in_channels", 20)
            OmegaConf.update(self.cfg, "train.batch_size", 100)
            OmegaConf.update(self.cfg, "train.epochs", 110)
            OmegaConf.update(self.cfg, "kernel.omega_0", 750.18)

    def _loading_pipeline(self):
        """
        1. Loading the dataset with transformations applied
        2. Split the dataset
        3. Save the processed dataset
        """
        x = torch.empty(34975, 16000, 1)
        y = torch.empty(34975, dtype=torch.long)
        batch_index = 0
        y_index = 0
        for foldername in (
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ):
            loc = os.path.join(self.data_dir, foldername)
            for filename in tqdm(os.listdir(loc)):
                audio, _ = torchaudio.load(
                    os.path.join(loc, filename),
                    channels_first=False,
                )

                # Discard samples shorter than 1 second
                if len(audio) != 16000:
                    continue

                transformed_audio = self.transform(audio)
                x[batch_index] = transformed_audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1

        # Split data into train, validation, and test sets
        total_size = len(y)
        train_size = int(self.train_split * total_size)  # 80% for training
        val_size = int(self.val_split * total_size)  # 10% for validation
        test_size = total_size - train_size - val_size  # Remaining 10% for testing

        train_data, val_data, test_data = random_split(
            list(zip(x, y)), [train_size, val_size, test_size]
        )

        # Convert splits to Hugging Face datasets
        self.dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "audio": [
                            d[0].tolist() for d in train_data
                        ],  # Convert tensors to lists
                        "label": [
                            d[1].item() for d in train_data
                        ],  # Convert tensors to scalars
                    }
                ),
                "val": Dataset.from_dict(
                    {
                        "audio": [d[0].tolist() for d in val_data],
                        "label": [d[1].item() for d in val_data],
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "audio": [d[0].tolist() for d in test_data],
                        "label": [d[1].item() for d in test_data],
                    }
                ),
            }
        )

        # Save to disk
        self.dataset.save_to_disk(self.serialized_dataset_path)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            SPEECHCOMMANDS(self.data_dir, download=True)

    def setup(self, stage: str):

        self._set_transform()

        # if already done load the preprocessed dataset
        if os.path.exists(self.serialized_dataset_path):
            print(f"Loading dataset from {self.serialized_dataset_path}...")
            self.dataset = DatasetDict.load_from_disk(self.serialized_dataset_path)
        else:
            # pipeline to load data
            self._loading_pipeline()

        # assign train/val datasets for use in dataloaders
        if stage == "fit":

            self.train_dataset, self.val_dataset = (
                self.dataset["train"],
                self.dataset["val"],
            )

        if stage == "test":
            self.test_dataset = self.dataset["test"]

        if stage == "predict":
            self.test_dataset = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":

    cfg = OmegaConf.load("config/config.yaml")

    dm = SpeechCommandsDataModule(
        cfg=cfg,
        data_dir="datasets",
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
