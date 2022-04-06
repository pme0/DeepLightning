from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule

from deeplightning.config.defaults import __config_groups__
from deeplightning.trainer.trainer import DLTrainer
from deeplightning.init.imports import init_module



def init_dataset(config: OmegaConf) -> LightningDataModule:
    """ Initialize LightningDataModule
    """
    s = config.data.module
    return init_module(short_config = s, config = config)


def init_model(config: OmegaConf) -> LightningModule:
    """ Initialize LightningModule
    """
    s = config.model.module
    return init_module(short_config = s, config = config)


def init_trainer(config: OmegaConf) -> DLTrainer:
    """ Initialize Deep Lightning Trainer
    """
    args = {
        "max_epochs": config.train.num_epochs,
        "gpus": config.engine.gpus,
        "num_nodes": config.engine.num_nodes,
        "strategy": config.engine.backend,
        "precision": config.engine.precision,
        "check_val_every_n_epoch": config.train.val_every_n_epoch,
        "log_every_n_steps": config.logger.log_every_n_steps,
        }
    return DLTrainer(config, args)


def init_everything(config: OmegaConf) -> Tuple[LightningModule, LightningDataModule, DLTrainer]:
    """ Initialize main modules
    """
    model = init_model(config)
    data = init_dataset(config)
    trainer = init_trainer(config)
    return model, data, trainer

