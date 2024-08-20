from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import datasets, transforms


class LRA(Dataset):

    def __init__(
        self, dataset_name, data_dir, download, train, batch_size, shuffle, transform
    ):
        """
        Initialization method for the long range arena dataset loader
        """
        super(LRA, self).__init__()
        self.dataset_name = dataset_name
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
        # self.dataset = load_dataset(self.dataset, split="train")
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
