import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from hydra import utils
from typing import Optional, Callable, Tuple, Dict, List, cast
from omegaconf import OmegaConf
import torchtext
from datasets import load_dataset, DatasetDict


class ListOpsDataModule(pl.LightningDataModule):
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
        self.dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        def listops_tokenizer(s):
            return s.translate(
                {ord("]"): ord("X"), ord("("): None, ord(")"): None}
            ).split()

        tokenizer = listops_tokenizer

        def tokenize(example):
            return {"tokens": tokenizer(example["Source"])[: self.max_length]}

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
        self._yaml_parameters()  # TODO set correct params

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])

        self.train_dataset, self.val_dataset, self.test_dataset = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            xs = torch.stack(
                [
                    torch.nn.functional.pad(
                        x,
                        [self.max_length - x.shape[-1], 0],
                        value=float(self.vocab["<pad>"]),
                    )
                    for x in xs
                ]
            )
            xs = xs.unsqueeze(1).float()
            ys = torch.tensor(ys)
            return xs, ys

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
