import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import datasets, transforms
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


class MFCC(Dataset):

    def __init__(self, data_dir, download, subset, batch_size, shuffle, transform):
        """
        Initialization method for the mfcc dataset loader
        """
        super(MFCC, self).__init__()
        self.data_dir = data_dir
        self.download = download
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transorm = transform

    def fetch_dataset(self):
        """
        Download the dataset from the internet source
        """
        self.dataset = SPEECHCOMMANDS(
            root=self.data_dir,
            subset=self.subset,
            download=self.download,
            transform=self.transform,
        )

        return self.dataset

    def preprocess(self):
        # TODO call from outside
        self.transform = T.MFCC(
            sample_rate=16000,  # Standard sample rate for Speech Commands
            n_mfcc=13,  # Number of MFCC features
            melkwargs={"n_mels": 23, "center": False},
        )

    def __len__(self):
        """
        Override the len() std method for this class objects if necessary
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Override the indexing std method for this class objects if necessary
        """
        return super().__getitem__(index)
