from pathlib import Path

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split
from collections import Counter
import string


from datasets import load_dataset, DatasetDict

from transformers import AutoTokenizer
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import requests
import tarfile
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from omegaconf import OmegaConf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")


from torchvision import transforms


class TextDataModule(pl.LightningDataModule):
    """
    Text Classification Dataset from LRA benchmarks 50000 movie reviews exit positive or negative
    """

    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
    ):

        super().__init__()

        self.data_dir = Path(data_dir) / "IMDB"
        self.num_workers = 0
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_imdb"
        )

        self.tokenizer_type = "word"
        self.special_tokens = ["<unk>", "<bos>", "<eos>"]

        self.max_length = 511
        self.val_split = 0.1

        self.cfg = cfg

        self._yaml_parameters()
        self.generator = torch.Generator(device=self.cfg.train.accelerator).manual_seed(42)


    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 60)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 2)
        OmegaConf.update(self.cfg, "kernel.omega_0", 2966.60)
        OmegaConf.update(self.cfg, "net.data_dim", 1)
        OmegaConf.update(self.cfg, "kernel.kernel_size", -1)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-5)
            OmegaConf.update(self.cfg, "train.learning_rate", 0.001)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)

        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.3)

    def prepare_data(self):
        # Download the dataset, if not already done
        self.dataset = load_dataset("imdb", cache_dir=self.data_dir)
        self.dataset = DatasetDict(
            train=self.dataset["train"], test=self.dataset["test"]
        )

    def _loading_pipeline(self):
        """
        1. Loading the dataset (train,val,test)
        2. Renaming and Word or Char Tokenization
        3. Building vocabulary
        4. Text Tokenization and Encoding
        5. Padding and Batching
        """

        def adapt_example(example):
            return {
                "Source": example["text"].lower(),
                "Target": example["label"],
            }

        self.dataset = self.dataset.map(
            adapt_example,
            remove_columns=["text", "label"],
            keep_in_memory=True,
            load_from_cache_file=False,
        )

        def w_tokenize(input):
            return {
                "Source": word_tokenize(input["Source"])[: self.max_length],
                "Target": input["Target"],
            }

        def char_tokenize(input):
            return {
                "Source": list(input["Source"])[: self.max_length],
                "Target": input["Target"],
            }

        if self.tokenizer_type == "word":
            tokenizer = w_tokenize
        elif self.tokenizer_type == "char":
            tokenizer = char_tokenize

        # Building vocabulary
        vocab_set = set()
        vocab_list = []

        for i, data in enumerate(self.dataset["train"]):
            examples = tokenizer(data)
            examples = examples["Source"]
            vocab_list.extend(examples)
            vocab_set.update(examples)  # add tokens to the vocabulary set

        vocab_set.update(self.special_tokens)  # special tokens
        vocab_set = list(set(vocab_set))
        token_counts = Counter(vocab_list)

        # Encoding
        word_to_number = {
            word: i + 1
            for i, word in enumerate(vocab_set)
            if token_counts[word] >= 15 or word in self.special_tokens
        }

        # Reserved tokens
        word_to_number["<pad>"] = 0
        word_to_number["<eos>"] = 1
        word_to_number["<unk>"] = -1

        def encode_tokens(input):
            tokens = input["Source"]
            encoded_tokens = [
                (
                    word_to_number[token]
                    if token in word_to_number
                    else word_to_number["<unk>"]
                )
                for token in tokens
            ] + [word_to_number["<eos>"]]

            if len(encoded_tokens) < self.max_length:
                padding_size = self.max_length - len(encoded_tokens)
                encoded_tokens = [0] * padding_size + encoded_tokens
            elif len(encoded_tokens) > self.max_length:
                encoded_tokens = encoded_tokens[: self.max_length]
            return {
                "Source": encoded_tokens,
                "Target": input["Target"],
            }

        # Tokenization
        self.dataset = self.dataset.map(
            tokenizer,
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        # Embeddings
        self.dataset = self.dataset.map(
            encode_tokens,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

        print(f"Saving dataset to {self.serialized_dataset_path}...")
        self.dataset.save_to_disk(self.serialized_dataset_path)

    def setup(self, stage):
        self.batch_size = self.cfg.train.batch_size

        # If already done, load the preprocessed dataset
        if os.path.exists(self.serialized_dataset_path):
            print(f"Loading dataset from {self.serialized_dataset_path}...")
            self.dataset = DatasetDict.load_from_disk(self.serialized_dataset_path)
        else:
            # Pipeline to load data
            self._loading_pipeline()

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(
                self.dataset["train"],
                [int(len(self.dataset["train"]) * (1 - self.val_split)),
                 int(len(self.dataset["train"]) * self.val_split)],
                generator=self.generator,
            )

        if stage == "test":
            self.test_dataset = self.dataset["test"]

        # Batching
        def collate_fn(batch):
            xs, ys = zip(*[(data["Source"], data["Target"]) for data in batch])

            # Efficiently convert list to tensor in one pass
            xs = [torch.tensor(x).view(1, self.max_length).unsqueeze(1).float() for x in xs]  # Combine unsqueeze and float conversion

            # Convert ys to tensor in one go
            ys = torch.tensor(ys)

            return xs, ys

        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )