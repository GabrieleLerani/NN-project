import torch
from torch.utils.data import random_split, DataLoader
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
        # download
        self.dataset = load_dataset("imdb", cache_dir=self.data_dir)
        self.dataset = DatasetDict(
            train=self.dataset["train"], test=self.dataset["test"]
        )
        if self.tokenizer_type == "word":
            self.tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.tokenizer_type == 'char'
            self.tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        max_length = self.max_length - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens": self.tokenizer(example["text"])[:max_length]
        }
        self.dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.dataset["train"]["tokens"],
            min_freq=self.vocab_min_freq,
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        self.vocab.set_default_index(self.vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": self.vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        self.dataset = self.dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )

    def setup(self, stage=None):

        self._set_transform()
        self._yaml_parameters()

        self.vocab_size = len(self.vocab)
        self.dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        self.train_dataset, self.test_dataset = (
            self.dataset["train"],
            self.dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
            # lengths = torch.tensor([len(x) for x in xs])
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

    # we define a separate DataLoader for each of train/val/test
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
