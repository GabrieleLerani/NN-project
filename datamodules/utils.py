from omegaconf import OmegaConf
import sklearn.model_selection
import torch
import os

def get_data_module(cfg : OmegaConf):
    
    assert cfg.data.dataset in [
        "s_mnist","p_mnist",
        "cifar10","s_cifar10","cifar100",
        "stl10",
        "speech_mfcc","speech_raw",
        "pathfinder","s_pathfinder",
        "listops",
        "image","s_image"
    ], "Dataset not supported"
    assert not (cfg.data.dataset in ["speech_mfcc","speech_raw","pathfinder","s_pathfinder","path_x","image","s_image"] and cfg.data.reduced_dataset), f"Reduced dataset not supported for {cfg.data.dataset}"

    # can be either sequential or permuted mnist
    if "mnist" in cfg.data.dataset: 
        from .mnist import MnistDataModule
        return MnistDataModule(cfg)
    elif "cifar10" in cfg.data.dataset:
        from .cifar10 import Cifar10DataModule
        return Cifar10DataModule(cfg)
    elif cfg.data.dataset == "cifar100":
        from .cifar100 import Cifar100DataModule
        return Cifar100DataModule(cfg)
    elif cfg.data.dataset == "stl10":
        from .stl10 import STL10DataModule
        return STL10DataModule(cfg)
    elif "speech" in cfg.data.dataset:
        from .speech import SpeechCommandsModule
        return SpeechCommandsModule(cfg)
    elif "pathfinder" in cfg.data.dataset:
        from .pathfinder import PathfinderDataModule
        return PathfinderDataModule(cfg)
    elif "image" in cfg.data.dataset:
        from .text import IMDBDataModule
        return IMDBDataModule(cfg)
    elif cfg.data.dataset == "listops":
        from .listops import ListOpsDataModule
        return ListOpsDataModule(cfg)
    
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
    if not os.path.exists(dir):
        os.makedirs(dir)
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir + "/" + tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir + "/" + filename))
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


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out