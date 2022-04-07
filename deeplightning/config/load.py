from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
from omegaconf import OmegaConf
import torch

from deeplightning.utilities.registry import __TaskRegistry__
from deeplightning.config.defaults import __ConfigGroups__
from deeplightning.utilities.messages import (info_message, 
                                              warning_message,
                                              error_message,
                                              config_print)


def load_config(config_file: str = "configs/base.yaml") -> OmegaConf:
    """ Load configuration from .yaml file.
    An updated artifact `config.yaml` is saved in `init_trainer()`
    to the logger's artifact storage path.
    """
    config = OmegaConf.load(config_file)
    config = configuration_defaults(config)
    config = configuration_checks(config)
    config_print(OmegaConf.to_yaml(config))
    return config


def configuration_defaults(user_config: OmegaConf) -> OmegaConf:
    """ Merge provided config with default config.
    The default parameters are overwritten if present
    in the user config provided.
    """

    # build default config
    default_config = OmegaConf.create()
    for g in __ConfigGroups__:
        config = OmegaConf.merge(default_config, OmegaConf.create(g))

    # merge default config with user connfig
    config = OmegaConf.merge(default_config, user_config)

    return config
    

def configuration_checks(config: OmegaConf) -> OmegaConf:
    """ Perform parameter checks and modify where inconsistent.
    """
    
    if config.task is None or config.task not in __TaskRegistry__:
        error_message(
            f"Task (config.task={config.task}) not in the registry "
            f"(__TaskRegistry__={__TaskRegistry__})."
        )
        raise ValueError

    if config.engine.gpus is not None:
        if not torch.cuda.is_available():
            warning_message(
                "GPUs selected but not available in this machine. "
                "Use 'engine.gpus: null' in the yaml config "
                "file to use CPU backend instead. 'engine.backend' "
                "can be set to 'null' or 'ddp'."
            )
    else:
        if config.engine.gpus is None and config.engine.backend is not None:
            warning_message(
                "No GPUs selected, therefore will overwrite "
                "config.engine.backend to use 'None' (CPU backend) "
                "(currently using backend '{}')".format(config.engine.backend)
            )
            config.engine.backend = None


    if config.engine.backend is not None:
        if "deepspeed" in config.engine.backend and \
            config.model.optimizer.type != "deepspeed.ops.adam.FusedAdam":
            warning_message(
                "PytorchLightning recommends FusedAdam optimizer "
                "when using DeepSpeed parallel backend "
                "(currently using '{}')".format(config.model.optimizer.type)
            )

    return config
   

