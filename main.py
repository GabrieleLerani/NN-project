import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from models import CCNN
from datamodules import get_data_module
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best_model",
        save_top_k=1,
        mode="min"
    )

    # early stopping callback
    early_stop_callback = EarlyStopping(monitor="val_loss")

    # create the trainer
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.epochs,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    # TODO retrieve the best model path after training if you want to use a trained model
    # checkpoint_callback.best_model_path


if __name__ == "__main__":
    main()
