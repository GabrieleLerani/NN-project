import pytorch_lightning as L
from torchvision.datasets import STL10
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import transforms
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


class STL10DataModule(L.LightningDataModule):
    def __init__(self, cfg, data_dir : str = "data/datasets"):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg
        self.num_workers = 7
        self.transform = transforms.ToTensor()


    def prepare_data(self):
        # download
        STL10(self.data_dir, split="train", download=True)
        STL10(self.data_dir, split="test", download=True)

        

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 64)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 3)
        OmegaConf.update(self.cfg, "net.out_channels", 10)
        OmegaConf.update(self.cfg, "net.data_dim", 2)
        OmegaConf.update(self.cfg, "kernel.omega_0", 954.28)
        OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.learning_rate", 0.02)            
            OmegaConf.update(self.cfg, "train.weight_decay", 0)
            
            
        elif hidden_channels == 380:
            OmegaConf.update(self.cfg, "train.learning_rate", 0.01)            
            OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
            
            

    def setup(self, stage: str):

        self._yaml_parameters()

        self.batch_size = self.cfg.train.batch_size

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
        return DataLoader(self.stl10_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.stl10_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.stl10_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.stl10_predict,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

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
    
    dm.setup("fit")

    dm.setup("test")
    
    dm.setup("predict")
    
    dm.show_samples(num_samples=2)