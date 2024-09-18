from .mnist import MnistDataModule
from .cifar10 import Cifar10DataModule
from .cifar100 import Cifar100DataModule
from .stl10 import STL10DataModule
from .speech import SpeechCommandsDataModule
from .img import ImageDataModule
from .text import TextDataModule
from .pathfinder import PathfinderDataModule
from .listops import ListOpsDataModule
from .utils import (
    get_data_module,
    feature_normalisation,
)
