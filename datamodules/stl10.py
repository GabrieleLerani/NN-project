import pytorch_lightning as L
from torchvision.datasets import STL10
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class STL10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/datasets/", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_channels = 3
        self.out_channels = 10

        self.transform = transforms.ToTensor()


    def prepare_data(self):
        # download
        STL10(self.data_dir, split="train", download=True)
        STL10(self.data_dir, split="test", download=True)



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            stl10_full = STL10(self.data_dir, split="train", transform=self.transform)
            self.stl10_train, self.stl10_val = random_split(
                stl10_full, [4500, 500], generator=torch.Generator().manual_seed(42)
            )
            print(f'Training set size: {len(self.stl10_train)}')
            print(f'Validation set size: {len(self.stl10_val)}')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.stl10_test = STL10(self.data_dir, split="test", transform=self.transform)
            print(f'Test set size: {len(self.stl10_test)}')

        if stage == "predict":
            self.stl10_predict = STL10(self.data_dir, split="test", transform=self.transform)
            print(f'Prediction set size: {len(self.stl10_predict)}')

    def train_dataloader(self):
        return DataLoader(self.stl10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.stl10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.stl10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.stl10_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

    def show_samples(self, num_samples: int = 5):
        dataset = STL10(self.data_dir, split="train", transform=self.transform)
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            image, label = dataset[i]
            image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()

if __name__ == "__main__":
    dm = STL10DataModule()
    dm.prepare_data()
    dm.setup("fit")

    dm.setup("test")
    
    dm.setup("predict")
    
    dm.show_samples(num_samples=2)