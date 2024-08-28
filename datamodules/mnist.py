import pytorch_lightning as L
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch


class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, type: str = "mnist"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_channels = 1
        self.out_channels = 10
        if type == "mnist":
            self.transform = transforms.ToTensor()
        if type == "smnist":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)) # flatten the image to 784 pixels
                  
            ])

        if type == "pmnist":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  
                transforms.Lambda(lambda x: x[torch.randperm(784)] )  # permutation of the 784 pixels
            ])


    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
            print(f'Training set size: {len(self.mnist_train)}')
            print(f'Validation set size: {len(self.mnist_val)}')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            print(f'Test set size: {len(self.mnist_test)}')

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False)
            print(f'Prediction set size: {len(self.mnist_predict)}')

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


if __name__ == "__main__":
    mnist = MnistDataModule("ckconv/data/datasets",32)
    mnist.prepare_data()
    mnist.setup("predict")
    
    