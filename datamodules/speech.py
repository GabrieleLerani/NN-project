import pytorch_lightning as L
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
from tqdm import tqdm
from torch.utils.data import random_split
from omegaconf import OmegaConf
from datasets import load_dataset, DatasetDict, Dataset
from utils import split_data, feature_normalisation

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
        self.selected_dirs = [
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
        ]

        assert self.type in ["speech_raw", "speech_mfcc"]
        self.cfg = cfg

        self._yaml_parameters()
        self._set_transform()

    def _set_transform(self):
        # 16 QAM modulation
        self.transform = torchaudio.transforms.Vol(
            gain=1.0 / 32768, gain_type="amplitude"
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
        X_list = []
        Y_list = []
        class_num = 0
        for foldername in self.selected_dirs:
            loc = os.path.join(self.speech_dir, foldername)
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load(
                    os.path.join(loc, filename), channels_first=False
                )

                if audio.size(0) != 16000:
                    continue

                # Apply transformation
                audio = self.transform(audio)

                # print(audio.shape)

                X_list.append(audio)
                Y_list.append(torch.tensor(class_num, dtype=torch.float32))
            class_num += 1

        # Concatenate lists into tensors
        X = torch.stack(X_list, dim=0)
        Y = torch.stack(Y_list, dim=0).unsqueeze(-1)

        # X shape (34975, 16000, 1) Y shape (34975,1)
        # batch-wise transforms

        if self.type == "speech_mfcc":
            # after squeeze shape of X is (34975, 16000) and then mfcc becomes (34975,20,time_frames)
            # where time_frames depends on the audio length and the parameters used (like n_fft and hop_length).
            X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64,hop_length=160)
            )(X.squeeze(-1)).detach()

            train_X, _, _ = split_data(X, Y)

            X = feature_normalisation(X, train_X)

        elif self.type == "speech_raw":
            # remove last dim and add the channle dim
            # the shape of X is (34975, 1, 16000)

            X = X.unsqueeze(1).squeeze(-1)
            train_X, _, _ = split_data(X, Y)
            X = feature_normalisation(X, train_X)

        train_data, val_data, test_data = split_data(X, Y)

        print(train_data.shape)

        # Convert splits to Hugging Face datasets
        self.dataset = DatasetDict(
            {
                "train": train_data,
                "val": val_data,
                "test": test_data,
            }
        )

        # TODO Save to disk--> slows down computation
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

    for batch in train_loader:
        print(f"Batch of images shape: {batch.shape}")
        break
