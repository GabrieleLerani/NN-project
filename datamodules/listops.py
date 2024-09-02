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


from torchvision import transforms


class ListOpsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
        max_length=512,  # Ensure this matches the model's max length
        append_bos=False,
        append_eos=True,
        tokenizer_name="bert-base-uncased",
    ):
        super().__init__()
        self.data_dir = Path(data_dir) / "datasets"
        self.num_workers = 7

        self.max_length = max_length
        self.append_bos = append_bos
        self.append_eos = append_eos

        self.tokenizer_name = tokenizer_name
        self.tokenizer = None

        # Determine data_type
        self.data_type = "sequence"
        self.data_dim = 1
        self.type = cfg.data.dataset
        self.cfg = cfg

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 10

        self._yaml_parameters()

    def prepare_data(self):

        if not self.data_dir.is_dir():
            self.download_and_extract_lra_release(self.data_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, use_fast=True
        )

        serialized_dataset_path = os.path.join(self.data_dir, "tokenized_dataset")

        if os.path.exists(serialized_dataset_path):
            print(f"Loading dataset from {serialized_dataset_path}...")
            self.dataset = DatasetDict.load_from_disk(serialized_dataset_path)
        else:

            dataset = load_dataset(
                "csv",
                data_files={
                    "train": str(
                        self.data_dir / "lra_release/listops-1000/basic_train.tsv"
                    ),
                    "val": str(
                        self.data_dir / "lra_release/listops-1000/basic_val.tsv"
                    ),
                    "test": str(
                        self.data_dir / "lra_release/listops-1000/basic_test.tsv"
                    ),
                },
                delimiter="\t",
                keep_in_memory=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, use_fast=True
            )

            # Adjust tokenizer for BOS and EOS tokens if needed
            if self.append_bos:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<bos>"]}
                )
            if self.append_eos:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<eos>"]}
                )

            tokenize = lambda example: {
                "tokens": self.tokenizer(
                    example["Source"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )["input_ids"]
                .squeeze()
                .tolist()  # Convert tensor to list
            }
            dataset = dataset.map(
                tokenize,
                remove_columns=["Source"],
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            def numericalize(example):
                tokens = (
                    (
                        self.tokenizer.convert_tokens_to_ids(["<bos>"])
                        if self.append_bos
                        else []
                    )
                    + example["tokens"]
                    + (
                        self.tokenizer.convert_tokens_to_ids(["<eos>"])
                        if self.append_eos
                        else []
                    )
                )
                return {"input_ids": tokens}

            dataset = dataset.map(
                numericalize,
                remove_columns=["tokens"],
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            self.dataset = dataset

            print(f"Saving dataset to {serialized_dataset_path}...")
            self.dataset.save_to_disk(serialized_dataset_path)

    def download_and_extract_lra_release(self, data_dir):

        if os.path.exists(Path(data_dir) / "lra_release"):
            print(
                f"Directory {data_dir} already exists. Skipping download and extraction."
            )
            return
        url = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
        local_filename = os.path.join(data_dir, "lra_release.gz")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Download the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract the tar.gz file
        with tarfile.open(local_filename, "r:gz") as tar:
            tar.extractall(path=data_dir)

        # Optionally, remove the tar.gz file after extraction
        os.remove(local_filename)

    def setup(self, stage=None):
        self._set_transform()

        self.batch_size = self.cfg.train.batch_size

        self.dataset.set_format(type="torch", columns=["input_ids", "Target"])

        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.dataset["train"],
            self.dataset["val"],
            self.dataset["test"],
        )

        def collate_batch(batch):
            input_ids = [data["input_ids"] for data in batch]
            labels = [
                data["Target"] for data in batch
            ]  # Ensure this matches your dataset column name

            # Retrieve the padding token ID from the tokenizer
            pad_value = self.tokenizer.pad_token_id

            # Convert lists of input IDs to tensors and pad them
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(seq) for seq in input_ids],
                batch_first=True,
                padding_value=pad_value,
            )

            # Truncate or pad to max_length
            if padded_input_ids.size(1) > self.max_length:
                padded_input_ids = padded_input_ids[
                    :, -self.max_length :
                ]  # Truncate to max_length
            else:
                padding_size = self.max_length - padded_input_ids.size(1)
                padded_input_ids = torch.nn.functional.pad(
                    padded_input_ids,
                    (0, padding_size),  # Pad on the right
                    value=pad_value,
                )

            input_tensor = padded_input_ids.float()
            label_tensor = torch.tensor(labels)

            return input_tensor, label_tensor

        # Set the collate function
        self.collate_fn = collate_batch

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


if __name__ == "__main__":
    pass


if __name__ == "__main__":
    dm = ListOpsDataModule(
        data_dir="./data",
        batch_size=32,
        test_batch_size=32,
        data_type="default",
    )
    dm.prepare_data()
    dm.setup()
    # Retrieve and print a sample
    train_loader = dm.train_dataloader()

    # Get a batch of data
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")

        # Print the first image and label in the batch
        print(f"First image tensor: {images[0]}")
        print(f"First label: {labels[0]}")

        # Break after first batch for demonstration
        break
