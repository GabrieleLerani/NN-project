import torch

from data import Loader
import torchvision.transforms as transforms


import hydra
from omegaconf import OmegaConf
from models import CCNN

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: OmegaConf) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    x = torch.ones(size=(8, cfg.net.in_channels, *(32,) * cfg.net.data_dim)).float()

    model = CCNN(
        in_channels=cfg.net.in_channels, # TODO in channels and out should be determined from dataset see datamodule
        out_channels=cfg.net.out_channels,
        data_dim=cfg.net.data_dim,
        cfg=cfg
    )

    x = model(x)

    print(x.shape)


if __name__ == "__main__":
    main()

    # params = yaml_utils.load_yaml(
    #     "/Users/applem2/Downloads/NN/NN-project/config/config.yaml"
    # )
    # device = (
    #     params["devices"]["cuda"]
    #     if torch.cuda.is_available()
    #     else (
    #         params["devices"]["mps"]
    #         if torch.backends.mps.is_available()
    #         else params["devices"]["cpu"]
    #     )
    # )
    # print(f"----- Using {device} device -----")

    # data_dir = params["data"]["data_dir"]
    # Loader().load(
    #     dataset="mnist",
    #     data_dir=data_dir,
    #     download=False,
    #     train=True,
    #     batch_size=10,
    #     shuffle=True,
    #     transform=transforms.Compose(
    #         [transforms.ToTensor()]  # Convert PIL images to tensors
    #     ),
    # )

    # model = GPCNN(L=1, in_channels=2, out_channels=2, num_classes=2).to(device)
    # print(f"----- Model structure: {model} -----\n\n")

    # for name, param in model.named_parameters():
    #     print(
    #         f"----- Layer: {name} | Size: {param.size()} | Values : {param[:2]} -----\n"
    #     )
