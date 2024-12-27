from typing import Any
from omegaconf import OmegaConf
import importlib

from deeplightning.utils.python_utils import exists, public_attributes


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
    return reference(**cfg.args) if exists(cfg.args) else reference()


def init_module(short_cfg: OmegaConf, cfg: OmegaConf) -> Any:
    """ Initialize module
    """
    reference = get_reference(short_cfg)
    instance = reference(cfg)
    return instance


def init_obj_from_config(cfg: OmegaConf, main_param: Any = None) -> Any:
    """ Initialize module from target (str) in config.
    """
    p = cfg.args
    if exists(p) and not isinstance(p, dict):
        p = public_attributes(p)

    reference = get_reference(cfg)

    if main_param is None:
        return reference(**p) if exists(p) else reference()
    else:
        return reference(main_param, **p) if exists(p) else reference(main_param)


def init_obj_from_target(target: str) -> Any:
    """ Initialize module from target (str).
    """
    lib, target = target.rsplit(".", 1)
    module = importlib.import_module(lib)
    return getattr(module, target)