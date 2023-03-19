from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule

from deeplightning.config.defaults import __ConfigGroups__
from deeplightning.trainer.trainer import DLTrainer
from deeplightning.init.imports import init_module
from deeplightning.utils.registry import (__MetricsRegistry__, 
                                          __LoggerRegistry__, 
                                          __HooksRegistry__)



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
        "gpus": cfg.engine.gpus,
        "num_nodes": cfg.engine.num_nodes,
        "strategy": cfg.engine.backend,
        "precision": cfg.engine.precision,
        "check_val_every_n_epoch": cfg.train.val_every_n_epoch,
        "log_every_n_steps": cfg.logger.log_every_n_steps,
        }
    return DLTrainer(cfg, args)


def init_logger(cfg: OmegaConf) -> None:
        """ Initialize Logger
        """

        # load logger
        logger = __LoggerRegistry__[cfg.logger.name](
            cfg = cfg,
            logged_metric_names = __HooksRegistry__[cfg.task]["LOGGED_METRICS_NAMES"]
        )

        # ensure all required attributes have been initialised
        attributes = ["run_id", "run_name", "run_dir", "artifact_path"]
        for attribute in attributes:
            if not hasattr(logger, attribute):
                raise AttributeError(f"Attribute '{attribute}' has not been set in DLLoger")
            
        return logger


def init_everything(cfg: OmegaConf) -> Tuple[LightningModule, LightningDataModule, DLTrainer]:
    """ Initialize main modules
    """
    model = init_model(cfg)
    data = init_dataset(cfg)
    trainer = init_trainer(cfg)
    return model, data, trainer


def init_metrics(cfg: OmegaConf) -> dict:
    """ Initialize performance metrics
    """
    return {k: v(cfg) for k, v in __MetricsRegistry__[cfg.task].items()}