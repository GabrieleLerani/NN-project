from utils import yaml_utils
import torch
from models import GPCNN
from data import Loader
import torchvision.transforms as transforms


if __name__ == "__main__":
    params = yaml_utils.load_yaml(
        "/Users/applem2/Downloads/NN/NN-project/config/config.yaml"
    )
    device = (
        params["devices"]["cuda"]
        if torch.cuda.is_available()
        else (
            params["devices"]["mps"]
            if torch.backends.mps.is_available()
            else params["devices"]["cuda"]
        )
    )
    print(f"----- Using {device} device -----")

    data_dir = params["data"]["data_dir"]
    Loader().load(
        dataset="mnist",
        data_dir=data_dir,
        download=False,
        train=True,
        batch_size=10,
        shuffle=True,
        transform=transforms.Compose(
            [transforms.ToTensor()]  # Convert PIL images to tensors
        ),
    )

    model = GPCNN(L=1, in_channels=2, out_channels=2, num_classes=2).to(device)
    print(f"----- Model structure: {model} -----\n\n")

    for name, param in model.named_parameters():
        print(
            f"----- Layer: {name} | Size: {param.size()} | Values : {param[:2]} -----\n"
        )
