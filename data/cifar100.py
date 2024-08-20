import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import datasets, transforms


class CIFAR100(Dataset):

    def __init__(self, data_dir, download, train, batch_size, shuffle, transform):
        """
        Initialization method for the cifar100 dataset loader
        """
        super(CIFAR100, self).__init__()
        self.data_dir = data_dir
        self.download = download
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transorm = transform

    def fetch_dataset(self):
        """
        Download the dataset from the internet source
        """
        self.dataset = datasets.CIFAR100(
            root=self.data_dir,
            download=self.download,
            train=self.train,
            transform=self.transform,
        )
        return self.dataset

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
