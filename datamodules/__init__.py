from .mnist import MnistDataModule
from .cifar10 import Cifar10DataModule
from .cifar100 import Cifar100DataModule
from .stl10 import STL10DataModule
from .utils import get_data_module, split_data, save_data, load_data_from_partition
# TODO add Pathfinder and Speech