from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
from omegaconf import OmegaConf
import importlib


def exists(x: ConfigElement) -> bool:
    return x is not None

    
def get_reference(config: OmegaConf) -> Any:
    """ Get a reference of the target class.
    """
    lib, target = config.type.rsplit(".", 1)
    module = importlib.import_module(lib)
    return getattr(module, target)


def get_instance(config: OmegaConf) -> Any:
    """ Get an instance of the target class.
    """
    reference = get_reference(config)
    return reference(**config.params) if exists(config.params) else reference()


def init_module(short_config: OmegaConf, config: OmegaConf) -> Any:
    """ Initialize module
    """
    reference = get_reference(short_config)
    instance = reference(config)
    return instance


def init_obj_from_config(config: OmegaConf, main_param: Any = None) -> Any:
    """ Initialize module from target (str) in config.
    """
    p = config.params
    reference = get_reference(config)
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