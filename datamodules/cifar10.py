import pytorch_lightning as L
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import transforms

class Cifar10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/datasets/", batch_size: int = 32, type: str = "cifar"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_channels = 3
        self.out_channels = 10

        if type == "cifar":
            self.transform = transforms.ToTensor()
        if type == "scifar":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)) # flatten the image to 1024 pixels
                  
            ])

        if type == "pcifar":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  
                transforms.Lambda(lambda x: x[torch.randperm(1024)] )  # permutation of the 784 pixels
            ])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar10_full = CIFAR10(self.data_dir, train=True)
            self.cifar10_train, self.cifar10_val = random_split(
                cifar10_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )
            print(f'Training set size: {len(self.cifar10_train)}')
            print(f'Validation set size: {len(self.cifar10_val)}')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            #print(f'Test set size: {len(self.cifar10_test)}')

        if stage == "predict":
            self.cifar10_predict = CIFAR10(self.data_dir, train=False)
            print(f'Prediction set size: {len(self.cifar10_predict)}')

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

if __name__ == "__main__":
    cifar10 = Cifar10DataModule("data/datasets",32)
    cifar10.prepare_data()
    cifar10.setup("fit")