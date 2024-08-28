import pytorch_lightning as L
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class Cifar100DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/datasets/", batch_size: int = 32, type: str = "cifar"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.type = type
        
        self.prepare_data()


    def prepare_data(self):
        # download
        train = CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

        # set out channels
        self.out_channels = len(train.classes)
        
        transform = transforms.Compose([transforms.ToTensor()])

        # set in channels
        self.in_channels = transform(train[0][0]).shape[0]
        
        # set size of the image
        self.size = train[0][0].size[0] * train[0][0].size[1]
        
        print(self.out_channels)

        self._set_transform()


    def _set_transform(self):

        if self.type == "cifar":
            self.transform = transforms.ToTensor()
        if self.type == "scifar":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)) # flatten the image to 1024 pixels
                  
            ])

        if self.type == "pcifar":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  
                transforms.Lambda(lambda x: x[torch.randperm(self.size)] )  # permutation of the 784 pixels
            ])



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.cifar100_full = CIFAR100(self.data_dir, train=True)
            self.cifar100_train, self.cifar100_val = random_split(
                self.cifar100_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
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

    def show_samples(self, num_samples: int = 5):
        dataset = self.cifar100_full
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            image, label = dataset[i]
            #image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()

if __name__ == "__main__":
    cifar100 = Cifar100DataModule()
    
    cifar100.setup("fit")

    cifar100.show_samples(3)