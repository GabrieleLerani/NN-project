import pytorch_lightning as L
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
from tqdm import tqdm
from torch.utils.data import random_split
from utils import normalise_data
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
            self.data_dir, "preprocessed_dataset_speech"
        )

        self.val_split = 0.1
        self.train_split = 0.9

        # Determine data_type
        self.type = cfg.data.dataset
        self.speech_dir = os.path.join(
            self.data_dir, "SpeechCommands/speech_commands_v0.02"
        )

        assert self.type in ["speech_raw", "speech_mfcc"]
        self.cfg = cfg

        self._yaml_parameters()

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
        ###################################################
        X = torch.empty(34975, 16000, 1)
        Y = torch.empty(34975, dtype=torch.long)
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
            loc = os.path.join(self.speech_dir, foldername)
            for filename in tqdm(os.listdir(loc)):
                audio, _ = torchaudio.load(
                    os.path.join(loc, filename),
                    channels_first=False,
                )

                audio = audio / 2**15  # normalization to have values in [0,1]

                # Discard samples shorter than 1 second
                if len(audio) != 16000:
                    continue
                X[batch_index] = audio
                Y[batch_index] = y_index
                batch_index += 1
            y_index += 1

        # X shape (34975, 16000, 1)

        if self.type == "sc_mfcc":
            # afetr squeeze shape of X is (34975, 16000) and then mfcc becomes (34975,20,time_frames)
            # where time_frames depends on the audio length and the parameters used (like n_fft and hop_length).
            X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(X.squeeze(-1)).detach()
            X = normalise_data(X.transpose(1, 2), Y).transpose(
                1, 2
            )  # transpose normalization and transpose back

        elif self.type == "sc_raw":
            # remove last dim and add the channle dim
            # the shape of X is (34975, 1, 16000)
            X = X.unsqueeze(1).squeeze(-1)
            X = normalise_data(X, Y)
        ###################################################
        # Split data into train, validation, and test sets
        total_size = len(Y)
        train_size = int(self.train_split * total_size)  # 80% for training
        val_size = int(self.val_split * total_size)  # 10% for validation
        test_size = total_size - train_size - val_size  # Remaining 10% for testing

        train_data, val_data, test_data = random_split(
            list(zip(X, Y)), [train_size, val_size, test_size]
        )

        # Convert splits to Hugging Face datasets
        self.dataset = DatasetDict(
            {
                "train": train_data,
                "val": val_data,
                "test": test_data,
            }
        )

        # TODO Save to disk
        # self.dataset.save_to_disk(self.serialized_dataset_path)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            print(f"Creating directory at {self.data_dir}")
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Directory created: {os.path.isdir(self.data_dir)}")

            try:
                self.data = SPEECHCOMMANDS(self.data_dir, download=True)
            except Exception as e:
                print(f"Error initializing SPEECHCOMMANDS: {e}")

    def setup(self, stage: str):

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
