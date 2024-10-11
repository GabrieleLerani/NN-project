from omegaconf import OmegaConf
import torch


def get_data_module(cfg: OmegaConf):

    assert cfg.data.dataset in [
        "s_mnist",
        "p_mnist",
        "cifar10",
        "s_cifar10",
        "cifar100",
        "stl10",
        "speech_mfcc",
        "speech_raw",
        "pathfinder",
        "s_pathfinder",
        "listops",
        "image",
        "s_image",
        "text",
    ], "Dataset not supported"
    assert not (
        cfg.data.dataset
        in [
            "speech_mfcc",
            "speech_raw",
            "pathfinder",
            "s_pathfinder",
            "path_x",
            "image",
            "s_image",
        ]
        and cfg.data.reduced_dataset
    ), f"Reduced dataset not supported for {cfg.data.dataset}"

    if "mnist" in cfg.data.dataset:
        from .mnist import MnistDataModule

        return MnistDataModule(cfg)
    elif "cifar100" in cfg.data.dataset:
        from .cifar100 import Cifar100DataModule

        return Cifar100DataModule(cfg)
    elif "cifar10" in cfg.data.dataset:
        from .cifar10 import Cifar10DataModule

        return Cifar10DataModule(cfg)
    elif "stl10" in cfg.data.dataset:
        from .stl10 import STL10DataModule

        return STL10DataModule(cfg)
    elif "speech" in cfg.data.dataset:
        from .speech import SpeechCommandsDataModule

        return SpeechCommandsDataModule(cfg)
    elif "pathfinder" in cfg.data.dataset:
        from .pathfinder import PathfinderDataModule

        return PathfinderDataModule(cfg)
    elif "image" in cfg.data.dataset:
        from .img import ImageDataModule

        return ImageDataModule(cfg)
    elif "text" in cfg.data.dataset:
        from .text import TextDataModule

        return TextDataModule(cfg)
    elif "listops" in cfg.data.dataset:
        from .listops import ListOpsDataModule

        return ListOpsDataModule(cfg)


def feature_normalisation(train_X):
    train_X_no_nan = torch.nan_to_num(train_X, nan=0.0)

    mean = torch.mean(train_X_no_nan, dim=0)
    std = torch.std(train_X_no_nan, dim=0)
    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0)

    X_normalized = (train_X - mean) / (std + 1e-5)

    return X_normalized
