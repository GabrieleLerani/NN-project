import pytorch_lightning as L
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import transforms

class Cifar100DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/datasets/", batch_size: int = 32, type: str = "cifar"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_channels = 3
        self.out_channels = 100

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
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar100_full = CIFAR100(self.data_dir, train=True)
            self.cifar100_train, self.cifar100_val = random_split(
                cifar100_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )
            print(f'Training set size: {len(self.cifar100_train)}')
            print(f'Validation set size: {len(self.cifar100_val)}')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar100_test = CIFAR100(self.data_dir, train=False, transform=self.transform)
            print(f'Test set size: {len(self.cifar100_test)}')

        if stage == "predict":
            self.cifar100_predict = CIFAR100(self.data_dir, train=False)
            print(f'Prediction set size: {len(self.cifar100_predict)}')

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar100_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

if __name__ == "__main__":
    cifar100 = Cifar100DataModule()
    cifar100.prepare_data()
    cifar100.setup("fit")
    cifar100.train_dataloader()
    cifar100.val_dataloader()

    cifar100.setup("test")
    cifar100.test_dataloader()

    cifar100.setup("predict")
    cifar100.predict_dataloader()