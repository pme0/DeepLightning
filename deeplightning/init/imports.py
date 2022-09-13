from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
from omegaconf import OmegaConf
import importlib


def exists(x: ConfigElement) -> bool:
    return x is not None

    
def get_reference(cfg: OmegaConf) -> Any:
    """ Get a reference of the target class.
    """
    lib, target = cfg.target.rsplit(".", 1)
    module = importlib.import_module(lib)
    return getattr(module, target)


def get_instance(cfg: OmegaConf) -> Any:
    """ Get an instance of the target class.
    """
    reference = get_reference(cfg)
    return reference(**cfg.params) if exists(cfg.params) else reference()


def init_module(short_cfg: OmegaConf, cfg: OmegaConf) -> Any:
    """ Initialize module
    """
    reference = get_reference(short_cfg)
    instance = reference(cfg)
    return instance


def init_obj_from_config(cfg: OmegaConf, main_param: Any = None) -> Any:
    """ Initialize module from target (str) in config.
    """
    p = cfg.params
    reference = get_reference(cfg)
    if main_param is None:
        instance = reference(**p) if exists(p) else reference()
    else:
        instance = reference(main_param, **p) if exists(p) else reference(main_param)
    return instance


def init_obj_from_target(target: str) -> Any:
    """ Initialize module from target (str).
    """
    lib, target = target.rsplit(".", 1)
    module = importlib.import_module(lib)
    return getattr(module, target)