import torch
from torch.utils.data import random_split, DataLoader
import torchtext.nn
from torchvision import transforms, datasets
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from hydra import utils
from typing import Optional, Callable, Tuple, Dict, List, cast
from omegaconf import OmegaConf
from datasets import load_dataset, DatasetDict
import torchtext


class IMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
        data_type="sequence",  # no other types allowed
    ):
        """
        Resolution in [32,64,128,256]
        Level in ["easy", "intermediate", "hard"]
        TODO look at the dir structure

        """
        super().__init__()
        self.data_dir = data_dir
        self.type = cfg.data.dataset
        # TODO set via conf or parameter init the parameters required
        self.cfg = cfg
        self.num_workers = 0  # for google colab training

        self.data_dir = Path(data_dir)

        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        self.input_channels = 1
        self.output_channels = 2

    def prepare_data(self):
        self.dataset = load_dataset("imdb", cache_dir=self.data_dir)
        self.dataset = DatasetDict(
            train=self.dataset["train"], test=self.dataset["test"]
        )

        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        def tokenize(example):
            return {"tokens": tokenizer(example["text"])[: self.max_length]}

        self.dataset = self.dataset.map(
            tokenize,
            remove_columns=["text"],
            num_proc=self.num_workers,
        )

        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.dataset["train"]["tokens"], min_freq=1, specials=["<pad>", "<unk>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])

        def numericalize(example):
            input_ids = self.vocab(example["tokens"])
            return {"input_ids": input_ids}

        self.dataset = self.dataset.map(
            numericalize,
            remove_columns=["tokens"],
            num_proc=self.num_workers,
        )

    def setup(self, stage=None):

        self._set_transform()
        self._yaml_parameters()

        self.vocab_size = len(self.vocab)
        self.dataset.set_format(type="torch", columns=["input_ids", "label"])

        self.train_dataset, self.test_dataset = (
            self.dataset["train"],
            self.dataset["test"],
        )

        def collate_batch(batch):
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["label"] for data in batch]

            pad_value = float(self.vocab["<pad>"])

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_value
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
                    value=pad_value,
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

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
        return test_dataloader


if __name__ == "__main__":
    pass
