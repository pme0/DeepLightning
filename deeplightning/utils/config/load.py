from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
import os
from omegaconf import OmegaConf
import torch

from deeplightning import TASK_REGISTRY
from deeplightning.utils.config.compute import runtime_compute
from deeplightning.utils.config.defaults import __ConfigGroups__
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
    #cfg = check_consistency(cfg)
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
    
    if cfg.task is None or cfg.task not in TASK_REGISTRY.get_element_names():
        error_message(
            f"Task (cfg.task={cfg.task}) not in the registry "
            f"(TASK_REGISTRY={TASK_REGISTRY.get_element_names()})."
        )
        raise ValueError
    
    if cfg.logger.name != "wandb":
        error_message(
            f"Logger (cfg.logger.name={cfg.logger.name}) not implemented."
        )
        raise NotImplementedError
    '''
    if cfg.engine.devices is not None:
        if not torch.cuda.is_available():
            warning_message(
                "GPUs {} selected but not available in this " 
                "machine. Will overwrite `engine.gpus` to use "
                "'auto' (CPU backend).".format(cfg.engine.devices)
            )
            cfg.engine.devices = "auto"
    else:
        if cfg.engine.devices is None and cfg.engine.strategy is not None:
            warning_message(
                "No GPUs selected, therefore will overwrite "
                "cfg.engine.strategy to use 'None' (CPU backend) "
                "(currently using backend '{}').".format(cfg.engine.strategy)
            )
            cfg.engine.strategy = None
    '''

    if cfg.engine.strategy is not None:
        if "deepspeed" in cfg.engine.strategy and \
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
    OmegaConf.save(cfg, f=os.path.join(path, "cfg.yaml"))