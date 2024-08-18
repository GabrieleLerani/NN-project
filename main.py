from utils import yaml_utils
import torch
from models import NN

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

    model = NN().to(device)
    print(f"----- Model structure: {model} -----\n\n")

    for name, param in model.named_parameters():
        print(
            f"----- Layer: {name} | Size: {param.size()} | Values : {param[:2]} -----\n"
        )
