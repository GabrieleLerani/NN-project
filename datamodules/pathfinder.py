import os
import tarfile
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import requests
from omegaconf import OmegaConf


# There's an empty file in the dataset
PATHFINDER_BLACKLIST = [
    "imgs/0/sample_172.png",
]
from PIL import UnidentifiedImageError


class PathfinderDataset(torch.utils.data.Dataset):
    """Pathfinder dataset created from a list of images."""

    def __init__(self, data_dir, typ, transform: Optional[Callable] = None) -> None:
        """
        Args:
            img_list (List[Tuple[str, int]]): List of tuples where each tuple contains
                an image path and its corresponding label.
            transform (Optional[Callable]): Optional transformation function or composition of transformations.
        """
        self.data_dir = data_dir
        self.type = typ
        self.transform = None

        self.img_list = self.create_imagelist()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple where the first element is the image tensor
                and the second element is the label.
        """
        img, label = self.img_list[idx]

        return img, label

    def create_imagelist(self) -> List[Tuple[str, int]]:

        # root dir where the image are placed
        directory = Path(self.data_dir).resolve()

        print(f"Loading images from {directory}")

        # metadata path where we get the class_idx
        path_list = sorted(
            list((directory / "metadata").glob("*.npy")),
            key=lambda path: int(path.stem),
        )
        instances = []
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            if self.type == "s_pathfinder":
                self.transform.transforms.append(transforms.Lambda(self.flatten_image))

        for metadata_file in path_list:
            with open(metadata_file, "r") as f:
                for metadata in f.read().splitlines():
                    # with tqdm(total=f.read().splitlines(), unit='files', desc='Extracting') as progress_bar:
                    # progress_bar.update(1)
                    metadata = metadata.split()
                    image_path = Path(metadata[0]) / metadata[1]
                    if (
                        str(image_path) in PATHFINDER_BLACKLIST
                        or str(metadata[0]) in PATHFINDER_BLACKLIST
                    ):
                        # print(f"Skipping {image_path}")
                        continue
                    image_path = Path(metadata[0]) / metadata[1]
                    full_image_path = directory / image_path  # Complete path to check

                    # Check if the image exists
                    if not full_image_path.exists():
                        # print(f"File does not exist, skipping: {full_image_path}")
                        continue  # Skip this iteration if the file does not exist

                    label = int(metadata[3])

                    img = Image.open(str(directory / image_path)).convert("L")

                    if self.transform:
                        img = self.transform(img)

                    instances.append((img, label))

        return instances

    @staticmethod
    def flatten_image(x):
        """Flattens the image tensor into a 1D tensor."""
        return x.view(1, -1)


class PathfinderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        data_dir: str = "datasets",
    ):
        super().__init__()

        level = "easy"
        resolution = "32"

        # if the light version (preprocessed by us) is used,
        # then level = easy and resolution = 32
        if cfg.data.light_lra:
            level = "easy"
            resolution = "32"

        level_dir = {
            "easy": "curv_baseline",
            "intermediate": "curv_contour_length_9",
            "hard": "curv_contour_length_14",
        }[level]
        self.global_dir = data_dir
        data_dir = data_dir + f"/lra_release/pathfinder{resolution}/{level_dir}"
        self.data_dir = Path(data_dir)
        self.type = cfg.data.dataset
        self.cfg = cfg

        self.val_split = 0.1
        self.test_split = 0.1

        self.num_workers = 0

        self._yaml_parameters()
        self.generator = torch.Generator(device=self.cfg.train.accelerator).manual_seed(42)

    def prepare_data(self):
        # download and extract only if the light lra version is not used
        if not self.cfg.data.light_lra:
            if not self.data_dir.is_dir():
                # Create data directory if it doesn't exist
                os.makedirs(self.data_dir, exist_ok=True)

            if not os.path.exists(Path(self.data_dir)):
                self.download_lra_release(self.data_dir)
                self._extract_lra_release()

            else:
                print("Zip already downloaded. Skipping download.")

    def _download_lra_release(self):
        url = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
        local_filename = os.path.join(self.data_dir, "lra_release.gz")

        # Download the file
        with requests.get(url, stream=True) as r:
            total = int(r.headers.get("content-length", 0))
            r.raise_for_status()
            with open(local_filename, "wb") as f, tqdm(
                desc=local_filename,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    def _extract_lra_release(self):
        local_filename = os.path.join(self.global_dir, "lra_release.gz")
        print(local_filename)

        # extraction the tar.gz file
        with tarfile.open(local_filename, "r:gz") as tar:
            # Get the total number of files in the tar archive
            total_files = len(tar.getmembers())

            # Initialize tqdm progress bar
            with tqdm(
                total=total_files, unit="files", desc="Extracting"
            ) as progress_bar:
                for member in tar.getmembers():
                    # Extract each file
                    tar.extract(member, path=self.data_dir)
                    # Update progress bar
                    progress_bar.update(1)

    def _extract_light_lra_release(self):
        local_filename = os.path.join(self.data_dir, "light_lra_release.tar.gz")

        # extraction the tar.gz file
        with tarfile.open(local_filename, "r:gz") as tar:
            # Get the total number of files in the tar archive
            total_files = len(tar.getmembers())

            # Initialize tqdm progress bar
            with tqdm(
                total=total_files, unit="files", desc="Extracting"
            ) as progress_bar:
                for member in tar.getmembers():
                    # Extract each file
                    tar.extract(member, path=self.data_dir)
                    # Update progress bar
                    progress_bar.update(1)

    def setup(self, stage=None):

        self.batch_size = self.cfg.train.batch_size

        self.dataset = PathfinderDataset(self.data_dir, typ=self.type)
        # compute lengths

        len_dataset = len(self.dataset)
        print(len_dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len

        # splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_len, val_len, test_len],
            generator=self.generator,
        )

    def _yaml_parameters(self):
        hidden_channels = self.cfg.net.hidden_channels

        OmegaConf.update(self.cfg, "train.batch_size", 50)
        OmegaConf.update(self.cfg, "train.epochs", 210)
        OmegaConf.update(self.cfg, "net.in_channels", 1)
        OmegaConf.update(self.cfg, "net.out_channels", 2)
        OmegaConf.update(self.cfg, "train.learning_rate", 0.01)

        if hidden_channels == 140:
            OmegaConf.update(self.cfg, "train.weight_decay", 0)

            if self.type == "pathfinder":
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 1239.14)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 33)

            elif self.type == "s_pathfinder":
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2272.56)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 33)

        elif hidden_channels == 380:

            if self.type == "pathfinder":
                OmegaConf.update(self.cfg, "net.data_dim", 2)
                OmegaConf.update(self.cfg, "train.weight_decay", 0)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.2)
                OmegaConf.update(self.cfg, "kernel.omega_0", 3908.32)
                OmegaConf.update(self.cfg, "kernel.kernel_size", 33)

            elif self.type == "s_pathfinder":
                OmegaConf.update(self.cfg, "net.data_dim", 1)
                OmegaConf.update(self.cfg, "train.weight_decay", 1e-6)
                OmegaConf.update(self.cfg, "train.dropout_rate", 0.1)
                OmegaConf.update(self.cfg, "kernel.omega_0", 2272.56)
                OmegaConf.update(self.cfg, "kernel.kernel_size", -1)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
            # persistent_workers=True,
            num_workers=self.num_workers,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_dataloader
