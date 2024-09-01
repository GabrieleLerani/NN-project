from pathlib import Path

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from datasets import load_dataset, DatasetDict

from transformers import AutoTokenizer


class IMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        data_type,
        max_length=4096,
        tokenizer_type="word",
        tokenizer_name="bert-base-uncased",
        vocab_min_freq=15,
        append_bos=False,
        append_eos=True,
        val_split=0.0,
    ):
        assert tokenizer_type in [
            "word",
            "char",
        ], f"tokenizer_type {tokenizer_type} not supported"

        super().__init__()

        # Save parameters to self
        self.data_dir = Path(data_dir) / "IMDB"
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = 7

        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        self.vocab_min_freq = vocab_min_freq
        self.append_bos = append_bos
        self.append_eos = append_eos
        self.val_split = val_split
        self.tokenizer_name = tokenizer_name

        # Determine data_type
        if data_type == "default":
            self.data_type = "sequence"
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 2

    def prepare_data(self):
        serialized_dataset_path = os.path.join(self.data_dir, "tokenized_dataset")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, use_fast=True
        )

        if os.path.exists(serialized_dataset_path):
            print(f"Loading dataset from {serialized_dataset_path}...")
            self.dataset = DatasetDict.load_from_disk(serialized_dataset_path)
        else:
            dataset = load_dataset("imdb", cache_dir=self.data_dir)

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, use_fast=True
            )
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<bos>", "<eos>"]}
            )

            def tokenize_function(example):
                encoding = self.tokenizer(
                    example["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"]
                    .squeeze()
                    .tolist()  # Convert tensor to list
                }

            # Tokenize and map to dataset
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=self.num_workers,
            )

            self.dataset = tokenized_datasets

            print(f"Saving dataset to {serialized_dataset_path}...")
            self.dataset.save_to_disk(serialized_dataset_path)

    def setup(self, stage=None):

        self._set_transform()
        self._yaml_parameters()

        self.dataset.set_format(type="torch", columns=["input_ids", "label"])

        self.train_dataset, self.test_dataset = (
            self.dataset["train"],
            self.dataset["test"],
        )

        # Use tokenizer padding token ID
        self.pad_token_id = self.tokenizer.pad_token_id

        def collate_batch(batch):
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["label"] for data in batch]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in input_ids],
                batch_first=True,
                padding_value=self.pad_token_id,
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
                    value=self.pad_token_id,
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
        pass

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
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
            collate_fn=self.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_dataloader


if __name__ == "__main__":
    pass


if __name__ == "__main__":

    dm = IMDBDataModule(
        data_dir="./data/datasets",
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
