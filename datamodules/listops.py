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
import nltk
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
import re
import numpy as np
import torch.nn as nn
from tqdm import tqdm

nltk.download("punkt_tab")

from torchvision import transforms


class ListOpsDataModule(pl.LightningDataModule):
    def __init__(self, cfg, data_dir: str = "datasets"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = 7
        self.batch_size = cfg.train.batch_size
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset_listops"
        )
        self.max_length = 5
        self.special_tokens = ["<unk>", "<bos>", "<eos>"]

        self.cfg = cfg

        self._yaml_parameters()

    def _set_transform(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 60)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "train.learning_rate", 0.001)
        OmegaConf.update(self.cfg, "kernel.omega_0", 784.66)
        OmegaConf.update(self.cfg, "net.data_dim", 1)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)

        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.25)

    def _download_lra_release(self):
        url = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
        local_filename = os.path.join(self.data_dir, "lra_release.gz")

        # Download the file
        with requests.get(url, stream=True) as r:
            total = int(r.headers.get("content-length", 0))
            r.raise_for_status()
            with open(local_filename, "wb") as f, tqdm(
                desc=local_filename,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    def _extract_lra_release(self):
        local_filename = os.path.join(self.data_dir, "lra_release.gz")

        # Extract the tar.gz file
        with tarfile.open(local_filename, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting"):
                tar.extract(member)

        # Optionally, remove the tar.gz file after extraction
        os.remove(local_filename)

    def _loading_pipeline(self):
        """
        1. Loading the dataset (train,val,test)
        dataset: {
            'Source': ['example text 1', 'example text 2', ...],
            'Target': [1, 0, ...]
            }
        2. Clean close brackets
        - Replace ] with X
        - Remove ( and )
        3. Whitespace Tokenization
        input: "AI is the future"
        output: ["AI", "is", "the","future"]
        4. Building vocabulary
         {"AI", "is", "the", "future"} (no repeated tokens)
        5. Text Tokenization and Encoding
        "AI" -> 3
        "is" -> 4
        "the" -> 5
        "future" -> 6
        [3,4,5,6]
        6. Padding adn Batching
        length = 5 --> [3,4,5,6,0]
        batch_size = 1 --> batch 1 --> {[3,4,5,6,0]}
        """

        # loading the datasets from tsv files
        self.dataset = load_dataset(
            "csv",
            data_files={
                "train": str(
                    self.data_dir / "lra_release/listops-1000/basic_train.tsv"
                ),
                "val": str(self.data_dir / "lra_release/listops-1000/basic_val.tsv"),
                "test": str(self.data_dir / "lra_release/listops-1000/basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )
        self.dataset.set_format(type="torch", columns=["Source", "Target"])

        def cleanFeatures(input):
            # Extract the source text
            text = input["Source"]

            # replace close brackets with 'X'
            text = re.sub(r"\]", "X", text)

            # remove open brackets '(' and close brackets ')'
            text = re.sub(r"[()]", "", text)

            return {"Source": word_tokenize(text), "Target": input["Target"]}

        # building vocabulary
        lengths = []
        vocab_set = set()
        for i, data in enumerate(self.dataset["train"]):
            examples = word_tokenize(data["Source"])
            examples = np.reshape(
                examples, (-1)
            ).tolist()  # flatten and convert to list
            lengths.append(len(examples))  # track the number of tokens
            vocab_set.update(examples)  # add tokens to the vocabulary set
        vocab_set.update(self.special_tokens)  # special tokens
        vocab_set = list(set(vocab_set))
        print(vocab_set)

        # encoding
        word_to_number = {word: i + 1 for i, word in enumerate(vocab_set)}

        def encode_tokens(input):
            tokens = input["Source"]
            encoded_tokens = [
                word_to_number[token] for token in tokens if token in word_to_number
            ]
            return {"Source": encoded_tokens, "Target": input["Target"]}

        for split in ["train", "val", "test"]:
            # apply clean close brackets
            self.dataset[split] = self.dataset[split].map(
                cleanFeatures,
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

    def prepare_data(self):
        if not self.cfg.data.light_lra:
            if not self.data_dir.is_dir():
                # Create data directory if it doesn't exist
                os.makedirs(self.data_dir, exist_ok=True)

            if not os.path.exists(Path(self.data_dir) / "lra_release"):
                self._download_lra_release()
                self._extract_lra_release()

            else:
                print("Zip already downloaded. Skipping download.")

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

        self.dataset.set_format(type="torch", columns=["Source", "Target"])

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

        # batching and padding
        def collate_batch(batch):
            input_ids = [data["Source"] for data in batch]
            labels = [data["Target"] for data in batch]

            # Convert lists of input IDs to tensors and pad them
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                [seq.clone().detach() for seq in input_ids],
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

    dm = ListOpsDataModule(
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
