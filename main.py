import torch
import hydra
import pytorch_lightning as pl
from kernel_logger import KernelLogger
from omegaconf import OmegaConf
from models import CCNN
from datamodules import get_data_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.profilers import PyTorchProfiler
import os

# In order for Hydra to generate again the files, go to config/config.yaml and look for defaults: and hydra:
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: OmegaConf) -> None:
    # 0. set device
    torch.set_default_device(cfg.train.accelerator)

    # 1. Create the dataset
    datamodule = get_data_module(cfg)
    
    print("################# Received configuration: #################")
    print(OmegaConf.to_yaml(cfg))
    
    # 2. Create the model
    model = create_model(cfg)
    # 3. Create the logger, callbacks, profiler and trainer
    logger, callbacks, profiler = setup_trainer_components(cfg)
    trainer = create_trainer(cfg, logger, callbacks, profiler)

    # 4. Train the model or use a pretrained one
    train(cfg, trainer, model, datamodule)
    
    

def create_model(cfg: OmegaConf) -> CCNN:
    return CCNN(
        in_channels=cfg.net.in_channels,
        out_channels=cfg.net.out_channels,
        data_dim=cfg.net.data_dim,
        cfg=cfg
    )

def setup_trainer_components(cfg: OmegaConf):
    # Setup logger
    logger = None
    filename = f"{cfg.data.dataset}_{cfg.net.no_blocks}_{cfg.net.hidden_channels}"
    if cfg.train.logger:
        logger = TensorBoardLogger("tb_logs", name=filename)
    
    # Setup callbacks
    callbacks = []
    if cfg.train.callbacks:
        # create checkpoints folder
        os.makedirs(f"checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", filename)
        os.makedirs(path, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            monitor="accuracy",
            dirpath=path,
            save_top_k=-1,
            every_n_epochs=1,
            mode="max"
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            patience=cfg.train.max_epoch_no_improvement, 
            verbose=True
        )
        model_summary_callback = ModelSummary(
            max_depth=-1
        )
        kernel_logger_callback = KernelLogger(
            filename
        )
        callbacks.extend([kernel_logger_callback, model_summary_callback, checkpoint_callback, early_stop_callback])

    # Setup profiler
    profiler = None
    if cfg.train.profiler:
        profiler = PyTorchProfiler(
            output_filename="profiler_output",
            group_by_input_shapes=True,
        )

    return logger, callbacks, profiler

def create_trainer(cfg: OmegaConf, logger: TensorBoardLogger, callbacks: list, profiler: PyTorchProfiler) -> pl.Trainer:
    return pl.Trainer(
        logger=logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.epochs,
        callbacks=callbacks,
        profiler=profiler,
        # TODO used for testing
        #limit_train_batches=3,
        #limit_val_batches=3,
        #limit_test_batches=3
    )

def train(cfg: OmegaConf, trainer: pl.Trainer, model: CCNN, datamodule) -> None:
    if not cfg.load_model.pre_trained:
        trainer.fit(model, datamodule)
    # Load the model from a checkpoint
    else:
        trainer.fit(model=model, train_dataloaders=datamodule, ckpt_path=get_checkpoint_path(cfg))

    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)


def get_checkpoint_path(cfg: OmegaConf) -> str:
    filename = f"{cfg.data.dataset}_{cfg.net.no_blocks}_{cfg.net.hidden_channels}"
    path = os.path.join("checkpoints", filename)
    checkpoint_path = os.path.join(path, f"epoch={cfg.load_model.epoch}-step={cfg.load_model.step}.ckpt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} file doesn't exist")
    return checkpoint_path

if __name__ == "__main__":
    main()
