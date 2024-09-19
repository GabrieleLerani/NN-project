from pathlib import Path

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split


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
        self.num_workers = 7
        self.batch_size = cfg.train.batch_size
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_imdb"
        )

        self.tokenizer_type = "word"
        self.special_tokens = ["<unk>", "<bos>", "<eos>"]
        self.max_length = 4096
        self.val_split = 0.1

        self.cfg = cfg

        self._yaml_parameters()

    def _set_transform(self):

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 60)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 2)
        OmegaConf.update(self.cfg, "kernel.omega_0", 2966.60)
        OmegaConf.update(self.cfg, "net.data_dim", 1)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-5)
            OmegaConf.update(self.cfg, "train.learning_rate", 0.001)
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
            return {"Source": example["text"], "Target": example["label"]}

        for split in ["train", "test"]:

            # apply renaming
            self.dataset[split] = self.dataset[split].map(
                adapt_example,
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

        def w_tokenize(input):
            return {"Source": word_tokenize(input["Source"]), "Target": input["Target"]}

        def char_tokenize(input):
            return {
                "Source": list(input["Source"]),  # or .encode("utf-8") byte level
                "Target": input["Target"],
            }

        if self.tokenizer_type == "word":
            tokenizer = w_tokenize
        elif self.tokenizer_type == "char":
            tokenizer = char_tokenize

        # building vocabulary

        lengths = []
        vocab_set = set()
        for i, data in enumerate(self.dataset["train"]):
            examples = tokenizer(data)
            examples = examples["Source"]
            examples = np.reshape(
                examples, (-1)
            ).tolist()  # flatten and convert to list
            lengths.append(len(examples))  # track the number of tokens
            vocab_set.update(examples)  # add tokens to the vocabulary set
        vocab_set.update(self.special_tokens)  # special tokens
        vocab_set = list(set(vocab_set))

        # encoding
        word_to_number = {word: i + 1 for i, word in enumerate(vocab_set)}

        def encode_tokens(input):
            tokens = input["Source"]
            encoded_tokens = [
                word_to_number[token] for token in tokens if token in word_to_number
            ]
            return {"Source": encoded_tokens, "Target": input["Target"]}

        for split in ["train", "test"]:

            # apply tokenizer
            self.dataset[split] = self.dataset[split].map(
                tokenizer,
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            # apply encoding
            self.dataset[split] = self.dataset[split].map(
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
            total_size = len(self.dataset["train"])
            val_size = int(self.val_split * total_size)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                self.dataset["train"], [train_size, val_size]
            )

        if stage == "test":
            self.test_dataset = self.dataset["test"]

        # batching and padding
        def collate_batch(batch):
            input_ids = [data["Source"] for data in batch]
            labels = [data["Target"] for data in batch]

            # Convert lists of input IDs to tensors and pad them
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq) for seq in input_ids],
                batch_first=True,
                padding_value=0,
            )

            # truncate or pad to max_length
            if padded_input_ids.size(1) > self.max_length:
                padded_input_ids = padded_input_ids[:, : self.max_length]
            else:
                padding_size = self.max_length - padded_input_ids.size(1)
                padded_input_ids = torch.nn.functional.pad(
                    padded_input_ids,
                    (0, padding_size),  # right padding
                    value=0,
                )

            input_tensor = padded_input_ids.float().unsqueeze(1)
            label_tensor = torch.tensor(labels)

            return input_tensor, label_tensor

        self.collate_fn = collate_batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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


if __name__ == "__main__":

    cfg = OmegaConf.load("config/config.yaml")

    dm = TextDataModule(
        cfg=cfg,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape} {labels.shape}")
        break
