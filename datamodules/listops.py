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


from torchvision import transforms


class ListOpsDataModule(pl.LightningDataModule):
    def __init__(self, cfg, data_dir: str = "data/datasets"):
        super().__init__()
        self.data_dir = Path(data_dir) / "datasets"
        self.num_workers = 7
        self.batch_size = self.cfg.train.batch_size
        self.serialized_dataset_path = os.path.join(
            self.data_dir, "preprocessed_dataset"
        )
        self.max_length = 512

        # Determine data_type
        self.type = cfg.data.type
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

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
            OmegaConf.update(self.cfg, "net.data_dim", 1)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
            OmegaConf.update(self.cfg, "kernel.omega_0", 2272.56)
        elif hidden_channels == 380:

            OmegaConf.update(self.cfg, "net.data_dim", 1)
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            OmegaConf.update(self.cfg, "train.dropout_rate", 0.25)
            OmegaConf.update(self.cfg, "kernel.omega_0", 2272.56)

    def _download_and_extract_lra_release(self):

        if os.path.exists(Path(self.data_dir) / "lra_release"):
            print(
                f"Directory {self.data_dir} already exists. Skipping download and extraction."
            )
            return
        url = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
        local_filename = os.path.join(self.data_dir, "lra_release.gz")

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Download the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract the tar.gz file
        with tarfile.open(local_filename, "r:gz") as tar:
            tar.extractall(path=self.data_dir)

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
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "listops-1000/basic_train.tsv"),
                "val": str(self.data_dir / "listops-1000/basic_val.tsv"),
                "test": str(self.data_dir / "listops-1000/basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )
        # TODO for each dataset in the dict

        def cleanFeatures(input):
            # Extract the source text
            text = input["Source"]

            # replace close brackets with 'X'
            text = re.sub(r"\]", "X", text)

            # remove open brackets '(' and close brackets ')'
            text = re.sub(r"[()]", "", text)

            return {"Source": text, "Target": input["Target"]}

        # whitespace tokenization
        w_tokenizer = WhitespaceTokenizer()

        def w_tokenize(input):
            return {
                "Source": w_tokenizer.tokenize(input["Source"]),
                "Target": input["Target"],
            }

        # building vocabulary
        lengths = []
        vocab_set = set()
        for i, data in enumerate(dataset["val"]):
            examples = data["Source"]
            examples = w_tokenizer.tokenize(examples)  # Tokenize the text
            examples = np.reshape(
                examples, (-1)
            ).tolist()  # Flatten and convert to list
            lengths.append(len(examples))  # Track the number of tokens
            vocab_set.update(examples)  # Add tokens to the vocabulary set
        vocab_set = list(set(vocab_set))

        # word tokenization
        nltk.download("punkt")

        def tokenize_text(input):
            return {"Source": word_tokenize(input["Source"]), "Target": input["Target"]}

        # encoding
        word_to_number = {word: i + 1 for i, word in enumerate(vocab_set)}

        def encode_tokens(example):
            tokens = example["Source"].split()
            encoded_tokens = [
                word_to_number[token] for token in tokens if token in word_to_number
            ]
            return {"Source": encoded_tokens, "Target": input["Target"]}

        for split in ["train", "val", "test"]:
            # apply clean close brackets
            dataset[split].map(
                cleanFeatures,
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            # apply whitespace tokenize
            dataset[split].map(
                w_tokenize,
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            # apply word tokenize
            dataset[split].map(
                tokenize_text,
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            # apply encoding
            dataset[split].map(
                encode_tokens,
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

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
                padded_input_ids = padded_input_ids[:, -self.max_length :]
            else:
                padding_size = self.max_length - padded_input_ids.size(1)
                padded_input_ids = torch.nn.functional.pad(
                    padded_input_ids,
                    (0, padding_size),  # right padding
                    value=0,
                )

            input_tensor = padded_input_ids.float()
            label_tensor = torch.tensor(labels)

            return input_tensor, label_tensor

        self.collate_fn = collate_batch

        self.dataset = dataset

        print(f"Saving dataset to {self.serialized_dataset_path}...")
        self.dataset.save_to_disk(self.serialized_dataset_path)

    def prepare_data(self):
        # download the dataset if not already done
        if not self.data_dir.is_dir():
            self._download_and_extract_lra_release(self.data_dir)

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

        self.dataset.set_format(type="torch", columns=["input_ids", "Target"])

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
            shuffle=True,
            num_workers=self.num_workers,
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

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":

    cfg = OmegaConf.load("config/config.yaml")

    dm = ListOpsDataModule(
        cfg=cfg,
        data_dir="data/datasets",
    )
    dm.prepare_data()
    # dm.setup()
    # Retrieve and print a sample
    # train_loader = dm.train_dataloader()

    # Get a batch of data
    # for images, labels in train_loader:
    #    print(f"Batch of images shape: {images.shape}")
    #    print(f"Batch of labels: {labels}")

    # Print the first image and label in the batch
#     print(f"First image tensor: {images[0]}")
#    print(f"First label: {labels[0]}")

# Break after first batch for demonstration
#   break
