import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from models import CCNN
from datamodules import get_data_module

# In order for Hydra to generate again the files, go to config/config.yaml and look for defaults: and hydra:
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: OmegaConf) -> None:
    

    # get the corresponding lightning datamodule
    datamodule = get_data_module(cfg)
    
    # create the ccnn model
    model = CCNN(
        in_channels=cfg.net.in_channels,
        out_channels=cfg.net.out_channels,
        data_dim=cfg.net.data_dim,
        cfg=cfg
    )

    # create the trainer
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.epochs
    )
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
