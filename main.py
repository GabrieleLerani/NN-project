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


# In order for Hydra to generate again the files, go to config/config.yaml and look for defaults: and hydra:
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: OmegaConf) -> None:
    # 1. Create the dataset
    datamodule = get_data_module(cfg)
    # 2. Create the model
    model = create_model(cfg)
    # 3. Create the logger, callbacks, profiler and trainer
    logger, callbacks, profiler = setup_trainer_components(cfg)
    trainer = create_trainer(cfg, logger, callbacks, profiler)

    print("################# Received configuration: #################")
    print(OmegaConf.to_yaml(cfg))
    # 4. Train the model or use a pretrained one
    if not cfg.pre_trained:
        train_and_evaluate(trainer, model, datamodule, callbacks)
    else:
        load_and_predict(trainer, model, datamodule, callbacks)

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
        checkpoint_callback = ModelCheckpoint(
            monitor="accuracy",
            dirpath="checkpoints",
            save_top_k=1,
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

def train_and_evaluate(trainer: pl.Trainer, model: CCNN, datamodule, callbacks: list) -> None:
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
    checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    print("Finished training, best model path: ", checkpoint_callback.best_model_path)

def load_and_predict(trainer: pl.Trainer, model: CCNN, datamodule, path: str) -> None:
    # TODO check how you take the best model path
    model = model.load_from_checkpoint(path)
    trainer.predict(model, datamodule)

if __name__ == "__main__":
    main()
