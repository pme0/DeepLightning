from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
from omegaconf import OmegaConf
import torch
from lightning import LightningModule, LightningDataModule

from deeplightning.utils.config.defaults import __ConfigGroups__
from deeplightning.utils.init.imports import init_module
from deeplightning.trainer.trainer import DLTrainer



def init_dataset(cfg: OmegaConf) -> LightningDataModule:
    """ Initialize LightningDataModule
    """
    s = cfg.data.module
    return init_module(short_cfg = s, cfg = cfg)


def init_model(cfg: OmegaConf) -> LightningModule:
    """ Initialize LightningModule
    """
    s = cfg.model.module
    return init_module(short_cfg = s, cfg = cfg)


def init_trainer(cfg: OmegaConf) -> DLTrainer:
    """ Initialize Deep Lightning Trainer
    """
    args = {
        "max_epochs": cfg.train.num_epochs,
        "num_nodes": cfg.engine.num_nodes,
        "accelerator": cfg.engine.accelerator,
        "strategy": cfg.engine.strategy,
        "devices": cfg.engine.devices,
        "precision": cfg.engine.precision,
        "check_val_every_n_epoch": cfg.train.val_every_n_epoch,
        "log_every_n_steps": cfg.logger.log_every_n_steps,
        }
    return DLTrainer(cfg, args)


def init_lightning_modules(cfg: OmegaConf) -> Tuple[LightningModule, LightningDataModule, DLTrainer]:
    """ Initialize main modules
    """
    model = init_model(cfg)
    data = init_dataset(cfg)
    trainer = init_trainer(cfg)
    return model, data, trainer