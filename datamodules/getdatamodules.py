from omegaconf import OmegaConf

def get_data_module(cfg : OmegaConf):
    
    assert cfg.data.dataset in ["smnist","pmnist","cifar10","scifar10","cifar100","stl10","mfcc","raw","pathfinder","path_x","image"], "Dataset not supported"

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


    