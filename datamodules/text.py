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

        # Save parameters to self
        self.data_dir = Path(data_dir) / "IMDB"
        self.num_workers = 1
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_imdb"
        )

        self.tokenizer_type = "char"
        self.special_tokens = ["<unk>", "<bos>", "<eos>"]

        self.max_length = 256  # real 4096
        self.val_split = 0.0

        self.cfg = cfg

        self._yaml_parameters()
        self.generator = torch.Generator(device=self.cfg.train.accelerator).manual_seed(
            42
        )
        self.batch_size = cfg.train.batch_size

    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 16)  # real 50
        OmegaConf.update(self.cfg, "train.epochs", 100)  # real 60
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 2)
        OmegaConf.update(self.cfg, "kernel.omega_0", 2966.60)
        OmegaConf.update(self.cfg, "net.data_dim", 1)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-5)
            OmegaConf.update(self.cfg, "train.learning_rate", 0.001)  # real 0.001
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)

        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            OmegaConf.update(self.cfg, "train.learning_rate", 0.02)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.3)

    def prepare_data(self):
        # download the dataset if not already done
        self.dataset = load_dataset("imdb", cache_dir=self.data_dir)
        self.dataset = DatasetDict(
            train=self.dataset["train"], test=self.dataset["test"]
        )
        import matplotlib.pyplot as plt

        # Step 1: Calculate the lengths of the text in the dataset
        lengths = [len(list(ex["text"])) for ex in self.dataset["train"]]

        # Step 2: Create a histogram
        plt.hist(lengths, bins=30, edgecolor='black')  # You can adjust the number of bins as needed

        # Step 3: Add titles and labels
        plt.title('Histogram of Text Lengths')
        plt.xlabel('Length of Text')
        plt.ylabel('Frequency')

        # Step 4: Show the plot
        plt.show()
        # print(f" MEDIUM LENGTH {sum(lengths) / len(lengths)}")
        # print(self.dataset["train"]["text"])

    def _loading_pipeline(self):
        """
        1. Loading the dataset (train,val,test)
        dataset: {
            'Source': ['example text 1', 'example text 2', ...],
            'Target': [1, 0, ...]
            }
        2. Renaming and Word or Char Tokenization
        ["AI", "is", "the", "future"] or {"A","I", "i","s" ...}
        3. Building vocabulary
         {"AI", "is", "the", "future"} (no repeated tokens)
        4. Text Tokenization and Encoding
        "AI" -> 3
        "is" -> 4
        "the" -> 5
        "future" -> 6
        [3,4,5,6]
        5. Padding adn Batching
        length = 5 --> [3,4,5,6,0]
        batch_size = 1 --> batch 1 --> {[3,4,5,6,0]}

        """

        def adapt_example(example):
            return {
                "Source": example["text"],
                "Target": example["label"],
            }

        self.dataset = self.dataset.map(
            adapt_example,
            remove_columns=["text", "label"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
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

        # building vocabulary
        import unicodedata

        # # Start with ascii letters, digits, and punctuation
        # allowed_characters = string.ascii_letters + string.punctuation + string.digits + string.whitespace

        # # Collect all unicode characters
        # all_unicode_characters = ''.join(chr(i) for i in range(0x110000) if unicodedata.category(chr(i)) not in ('Cc', 'Cf', 'Cn', 'Cs'))

        # # Combine them all
        # allowed_characters += all_unicode_characters

        # # Now you have all characters combined (ASCII + Unicode)
        # print(allowed_characters)

        # word_to_number = {char: i + 2 for i, char in enumerate(allowed_characters)}

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

        # encoding
        word_to_number = {
            word: i + 1
            for i, word in enumerate(vocab_set)
            if token_counts[word] >= 15 or word in self.special_tokens
        }

        # reserved tokebns
        word_to_number["<pad>"] = 0
        word_to_number["<eos>"] = 1
        word_to_number["<unk>"] = -1

        print(word_to_number)

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

        # tokenization
        self.dataset = self.dataset.map(
            tokenizer,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        # embeddings
        self.dataset = self.dataset.map(
            encode_tokens,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )

        print(f"Saving dataset to {self.serialized_dataset_path}...")
        self.dataset.save_to_disk(self.serialized_dataset_path)

    def setup(self, stage):

        # if already done load the preprocessed dataset
        if os.path.exists(self.serialized_dataset_path):
            print(f"Loading dataset from {self.serialized_dataset_path}...")
            self.dataset = DatasetDict.load_from_disk(self.serialized_dataset_path)
        else:
            # pipeline to load data
            self._loading_pipeline()

        # set the relevant
        self._set_transform()

        # assign train/val datasets for use in dataloaders
        if stage == "fit":

            self.train_dataset = self.dataset["train"]
            self.val_dataset = self.dataset["test"]

        if stage == "test":
            self.test_dataset = self.dataset["test"]

        # batching and padding
        def collate_fn(batch):
            xs, ys = zip(*[(data["Source"], data["Target"]) for data in batch])
            xs = torch.stack([torch.tensor(x) for x in xs])
            xs = xs.unsqueeze(1).float()
            ys = torch.tensor(ys)
            return xs, ys

        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
            # drop_last=True,
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


if __name__ == "__main__":

    cfg = OmegaConf.load("config/config.yaml")

    dm = TextDataModule(
        cfg=cfg,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    # for batch in train_loader:
    #     input_tensor, label_tensor = batch
    #     # print(f"input {input_tensor}clabel {label_tensor}")
    #     print(f"input single  {input_tensor[0]} label single {label_tensor[0]}")
    # print(f"Input shape: {input_tensor.shape}, Labels shape: {label_tensor.shape}")
