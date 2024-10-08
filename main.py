import torch
import hydra
import pytorch_lightning as pl
from kernel_logger import KernelLogger
from omegaconf import OmegaConf
from models import CCNN
from datamodules import get_data_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
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
        cfg=cfg,
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

        path_top = os.path.join(path, "top")
        os.makedirs(path_top, exist_ok=True)
        checkpoint_top_callback = ModelCheckpoint(
            monitor="accuracy",
            dirpath=path_top,
            save_top_k=1,
            every_n_epochs=1,
            mode="max",
        )
        path_last = os.path.join(path, "last")
        os.makedirs(path_last, exist_ok=True)
        checkpoint_last_callback = ModelCheckpoint(
            dirpath=path_last,
            every_n_epochs=1,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.max_epoch_no_improvement,
            verbose=True,
        )
        model_summary_callback = ModelSummary(max_depth=-1)
        kernel_logger_callback = KernelLogger(filename)
        learning_rate_callback = LearningRateMonitor(logging_interval="step")
        callbacks.extend(
            [
                kernel_logger_callback,
                model_summary_callback,
                checkpoint_top_callback,
                checkpoint_last_callback,
                early_stop_callback,
                learning_rate_callback,
            ]
        )

    # Setup profiler
    profiler = None
    if cfg.train.profiler:
        profiler = PyTorchProfiler(
            output_filename="profiler_output",
            group_by_input_shapes=True,
        )

    return logger, callbacks, profiler


def create_trainer(
    cfg: OmegaConf,
    logger: TensorBoardLogger,
    callbacks: list,
    profiler: PyTorchProfiler,
) -> pl.Trainer:
    # Ensure ModelCheckpoint is in the callbacks
    if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        raise ValueError(
            "ModelCheckpoint callback must be included in the callbacks list."
        )

    return pl.Trainer(
        logger=logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.epochs,
        callbacks=callbacks,
        profiler=profiler,
        # TODO used for testing
        # limit_train_batches=30,
        # limit_val_batches=30,
        # limit_test_batches=30
    )


def train(cfg: OmegaConf, trainer: pl.Trainer, model: CCNN, datamodule) -> None:
    # start new training
    if not cfg.load_model.pre_trained or not exists_checkpoint_path(cfg):
        trainer.fit(model=model, datamodule=datamodule)
    # Load the model from a checkpoint
    else:
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=get_checkpoint_path(cfg)
        )

    trainer.validate(
        model=model, datamodule=datamodule, ckpt_path=get_checkpoint_path(cfg)
    )
    trainer.test(model=model, datamodule=datamodule, ckpt_path=get_checkpoint_path(cfg))


def get_checkpoint_path(cfg: OmegaConf) -> str:
    filename = f"{cfg.data.dataset}_{cfg.net.no_blocks}_{cfg.net.hidden_channels}"
    path = os.path.join("checkpoints", filename)
    if cfg.load_model.model == "last":
        path = os.path.join(path, "last")
    elif cfg.load_model.model == "top":
        path = os.path.join(path, "top")
    # there should be only one file
    files = os.listdir(path)
    checkpoint_path = os.path.join(path, files[0])

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} file doesn't exist")
    return checkpoint_path


def exists_checkpoint_path(cfg: OmegaConf) -> str:
    filename = f"{cfg.data.dataset}_{cfg.net.no_blocks}_{cfg.net.hidden_channels}"
    path = os.path.join("checkpoints", filename)
    if cfg.load_model.model == "last":
        path = os.path.join(path, "last")
    elif cfg.load_model.model == "top":
        path = os.path.join(path, "top")
    # there should be only one file
    files = os.listdir(path)
    if len(files) > 0:
        return True
    return False


if __name__ == "__main__":
    main()
