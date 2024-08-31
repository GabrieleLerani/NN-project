import pytorch_lightning as L
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, TensorDataset
from .utils import split_data, load_data_from_partition, save_data, normalise_data
import torch
import os
from tqdm import tqdm
from omegaconf import OmegaConf

class SpeechCommandsModule(L.LightningDataModule):
    def __init__(self, cfg, data_dir : str = "datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.data_processed_location = self.data_dir + "/SpeechCommands/processed_data"
        self.download_location = self.data_dir + "/SpeechCommands/speech_commands_v0.02"
        self.type = cfg.data.dataset
        self.cfg = cfg
        self.num_workers = 0 # for google colab training


    def process_data(self):
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
            loc = self.download_location + "/" + foldername
            for filename in tqdm(os.listdir(loc)):
                audio, _ = torchaudio.load(
                    loc + "/" + filename,
                    channels_first=False,
                )

                # A few samples are shorter than the full length; for simplicity we discard them.
                if len(audio) != 16000:
                    continue

                x[batch_index] = audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1


        # If MFCC, then we compute these coefficients.
        if self.type == "sc_mfcc":
            x = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(x.squeeze(-1)).detach()
            # X is of shape (batch=34975, channels=20, length=161)
        else:
            x = x.unsqueeze(1).squeeze(-1)
            # X is of shape (batch=34975, channels=1, length=16000)

        # Normalize data
        if self.type == "sc_mfcc":
            x = normalise_data(x.transpose(1, 2), y).transpose(1, 2)
        else:
            x = normalise_data(x, y)

        train_x, val_x, test_x = split_data(x, y)
        train_y, val_y, test_y = split_data(y, y)

        
        return (
            train_x,
            val_x,
            test_x,
            train_y,
            val_y,
            test_y,
        )


    def prepare_data(self):
        # download
        SPEECHCOMMANDS(self.data_dir, download=True)
        if not os.path.exists(self.data_processed_location + "/train_x.pt"):
            train_x, val_x, test_x, train_y, val_y, test_y = self.process_data()

            save_data(
                self.data_processed_location,
                train_x=train_x,
                val_x=val_x,
                test_x=test_x,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y,
            )

    
    def _yaml_parameters(self):
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 1)
        OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
        OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
        OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)

        # 140 and 380 hidden_channels have same parameters
        if self.type == "sc_raw":
            OmegaConf.update(self.cfg, "net.in_channels", 1)
            OmegaConf.update(self.cfg, "train.batch_size", 20)
            OmegaConf.update(self.cfg, "train.epochs", 160)
            OmegaConf.update(self.cfg, "kernel.omega_0", 1295.61)
        elif self.type == "sc_mfcc":
            OmegaConf.update(self.cfg, "net.in_channels", )
            OmegaConf.update(self.cfg, "train.batch_size", 100)
            OmegaConf.update(self.cfg, "train.epochs", 110)
            OmegaConf.update(self.cfg, "kernel.omega_0", 750.18)


    def setup(self, stage: str):
        self._yaml_parameters()

        self.batch_size = self.cfg.train.batch_size

        if stage == "fit":
            # train
            x_train, y_train = load_data_from_partition(
                self.data_processed_location, partition="train"
            )
            self.train_dataset = TensorDataset(x_train, y_train)
            # validation
            x_val, y_val = load_data_from_partition(
                self.data_processed_location, partition="val"
            )
            self.val_dataset = TensorDataset(x_val, y_val)
        if stage == "test":
            # test
            x_test, y_test = load_data_from_partition(
                self.data_processed_location, partition="test"
            )
            self.test_dataset = TensorDataset(x_test, y_test)
        if stage == "predict":
            # predict
            x_test, y_test = load_data_from_partition(
                self.data_processed_location, partition="test"
            )
            self.test_dataset = TensorDataset(x_test, y_test)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    sc = SpeechCommandsModule(cfg)

    sc.prepare_data()
    sc.setup("fit")
    sc.setup("test")