import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from models import CCNN
from datamodules import MnistDataModule

# In order for Hydra to generate again the files, go to config/config.yaml and look for defaults: and hydra:
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: OmegaConf) -> None:
    
    x = torch.randn(64, 3, 32, 32)

    model = CCNN(
        in_channels=cfg.net.in_channels, # TODO in channels and out should be determined from dataset see datamodule
        out_channels=cfg.net.out_channels,
        data_dim=cfg.net.data_dim,
        cfg=cfg
    )

    print(model(x).shape)



    # datamodule = MnistDataModule("ckconv/data/datasets", 32, "smnist")

    # trainer = pl.Trainer(accelerator=cfg.train.accelerator, devices=cfg.train.devices, min_epochs=1, max_epochs=cfg.train.epochs)
    # trainer.fit(model, datamodule)
    # trainer.validate(model, datamodule)
    # trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
