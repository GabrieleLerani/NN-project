import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from data import MnistLoader


class Loader:

    def __init__(self):
        """
        Method used to init the general loader class used to load different datasets
        """
        pass

    def load(self, dataset, data_dir, download, train, batch_size, shuffle, transform):
        """
        Method used to load the specific dataset
        """
        dataset_specific_loader = self.select_dataset_loader(
            dataset=dataset,
            data_dir=data_dir,
            download=download,
            train=train,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=transform,
        )
        t_data = dataset_specific_loader.fetch_dataset()
        dataloader = DataLoader(t_data, batch_size=batch_size)

        for X, y in dataloader:
            print(f"Shape of X: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        raise NotImplementedError("This part is not implemented yet")

    def select_dataset_loader(
        self, dataset, data_dir, download, train, batch_size, shuffle, transform
    ):
        """
        Loading the specific dataset from all the possible datasets available:
            - mnist
            - cifar10
            - cifar100
            - stl10
            - imdb
            - modelnet
        """
        if dataset == "mnist":
            dataset_loader = MnistLoader(
                data_dir=data_dir,
                download=download,
                train=train,
                batch_size=batch_size,
                shuffle=shuffle,
                transform=transform,
            )
        elif dataset == "cifar10":
            raise NotImplementedError("This part for cifar10 is not implemented yet")
        elif dataset == "cifar100":
            raise NotImplementedError("This part for cifar100 is not implemented yet")
        elif dataset == "stl10":
            raise NotImplementedError("This part for stl10 is not implemented yet")
        elif dataset == "imdb":
            raise NotImplementedError("This part for imdb is not implemented yet")
        elif dataset == "modelnet":
            raise NotImplementedError("This part for modelnet is not implemented yet")
        else:
            raise ValueError(
                """Dataset not found the list of possible dataset is: 
                            - mnist
                            - cifar10
                            - cifar100
                            - stl10
                            - imdb
                            - modelnet
                """
            )
        return dataset_loader
