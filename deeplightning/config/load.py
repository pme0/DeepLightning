from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
import os
from omegaconf import OmegaConf
import torch

from deeplightning.config.compute import runtime_compute
from deeplightning.utils.registry import (__TaskRegistry__, 
                                          __LoggerRegistry__,)
from deeplightning.config.defaults import __ConfigGroups__
from deeplightning.utils.messages import (info_message, 
                                          warning_message,
                                          error_message,
                                          config_print,)


def load_config(config_file: str = "configs/base.yaml") -> OmegaConf:
    """ Load configuration from .yaml file.
    An updated artifact `cfg.yaml` is saved in `init_trainer()`
    to the logger's artifact storage path.
    """
    cfg = OmegaConf.load(config_file)
    cfg = merge_defaults(cfg)
    cfg = check_consistency(cfg)
    cfg = runtime_compute(cfg)
    OmegaConf.resolve(cfg)
    #config_print(OmegaConf.to_yaml(cfg))
    return cfg


def merge_defaults(user_config: OmegaConf) -> OmegaConf:
    """ Merge provided config with default config.
    The default parameters are overwritten if present
    in the user config provided.
    """

    # build default config
    default_config = OmegaConf.create()
    for g in __ConfigGroups__:
        cfg = OmegaConf.merge(default_config, OmegaConf.create(g))

    # merge default config with user connfig
    cfg = OmegaConf.merge(default_config, user_config)

    return cfg
    

def check_consistency(cfg: OmegaConf) -> OmegaConf:
    """ Perform parameter checks and modify where inconsistent.
    """
    
    if cfg.task is None or cfg.task not in __TaskRegistry__:
        error_message(
            f"Task (cfg.task={cfg.task}) not in the registry "
            f"(__TaskRegistry__={__TaskRegistry__})."
        )
        raise ValueError
    
    if cfg.logger.name not in __LoggerRegistry__:
        error_message(
            f"Logger (cfg.logger.name={cfg.logger.name}) not in the registry "
            f"(__LoggerRegistry__={__LoggerRegistry__})."
        )
        raise ValueError

    if cfg.engine.gpus is not None:
        if not torch.cuda.is_available():
            warning_message(
                "GPUs {} selected but not available in this " 
                "machine. Will overwrite `engine.gpus` to use "
                "'None' (CPU backend).".format(cfg.engine.gpus)
            )
            cfg.engine.gpus = None
    else:
        if cfg.engine.gpus is None and cfg.engine.backend is not None:
            warning_message(
                "No GPUs selected, therefore will overwrite "
                "cfg.engine.backend to use 'None' (CPU backend) "
                "(currently using backend '{}').".format(cfg.engine.backend)
            )
            cfg.engine.backend = None


    if cfg.engine.backend is not None:
        if "deepspeed" in cfg.engine.backend and \
            cfg.model.optimizer.target != "deepspeed.ops.adam.FusedAdam":
            warning_message(
                "PytorchLightning recommends FusedAdam optimizer "
                "when using DeepSpeed parallel backend "
                "(currently using '{}')".format(cfg.model.optimizer.target)
            )

    return cfg


def log_config(cfg: OmegaConf, path: str) -> None:
    """ Save configuration (.yaml)
    """
    if not OmegaConf.is_config(cfg):
        error_message(
            "Attempting to save a config artifact but the object "
            "provided is not of type omegaconf.dictconfig.DictConfig.")
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    OmegaConf.save(cfg, f = os.path.join(path, "cfg.yaml"))