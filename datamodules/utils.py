from omegaconf import OmegaConf
import sklearn
import torch
import os

def get_data_module(cfg : OmegaConf):
    
    assert cfg.data.dataset in ["smnist","pmnist","cifar10","scifar10","cifar100","stl10","sc_mfcc","sc_raw","pathfinder","path_x","image"], "Dataset not supported"

    # can be either sequential or permuted mnist
    if "mnist" in cfg.data.dataset: 
        from .mnist import MnistDataModule
        return MnistDataModule(cfg)
    if cfg.data.dataset == "cifar10" or cfg.data.dataset == "scifar10":
        from .cifar10 import Cifar10DataModule
        return Cifar10DataModule(cfg)
    if cfg.data.dataset == "cifar100":
        from .cifar100 import Cifar100DataModule
        return Cifar100DataModule(cfg)
    if cfg.data.dataset == "stl10":
        from .stl10 import STL10DataModule
        return STL10DataModule(cfg)
    
    # TODO other dataset

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def load_data_from_partition(data_loc, partition):
    assert partition in ["train", "val", "test"]
    # load tensors
    tensors = load_data(data_loc)
    # select partition
    name_x, name_y = f"{partition}_x", f"{partition}_y"
    x, y = tensors[name_x], tensors[name_y]
    return x, y
