import os

from omegaconf import OmegaConf, DictConfig, ListConfig

from deeplightning import TASK_REGISTRY
from deeplightning.utils.config.compute import runtime_compute
from deeplightning.utils.config.defaults import __ConfigGroups__
from deeplightning.utils.messages import (
    info_message,
    warning_message,
    error_message,
)


def load_config(config_file: str = "configs/base.yaml") -> DictConfig:
    cfg = OmegaConf.load(config_file)
    cfg = merge_defaults(cfg)
    #cfg = check_consistency(cfg)
    cfg = runtime_compute(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_config(cfg: DictConfig) -> DictConfig:
    cfg = merge_defaults(cfg)
    cfg = expand_home_directories(cfg)
    #cfg = check_consistency(cfg)
    #cfg = runtime_compute(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def expand_home_directories(cfg: DictConfig) -> DictConfig:
    """Expand home directories specified with '~/' into absolute paths."""
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            cfg[key] = expand_home_directories(value)
    elif isinstance(cfg, ListConfig):
        for index, item in enumerate(cfg):
            cfg[index] = expand_home_directories(item)
    elif isinstance(cfg, str):
        if cfg.startswith("~/"):
            return os.path.expanduser(cfg)
    return cfg


def merge_defaults(user_config: DictConfig) -> DictConfig:
    """Merge provided config with default config.
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
    """Perform parameter checks and modify where inconsistent.
    """
    
    if cfg.task is None or cfg.task not in TASK_REGISTRY.get_element_names():
        error_message(
            f"Task (cfg.task={cfg.task}) not in the registry "
            f"(TASK_REGISTRY={TASK_REGISTRY.get_element_names()})."
        )
        raise ValueError
    
    if cfg.logger.provider != "wandb":
        error_message(
            f"Logger (cfg.logger.provider={cfg.logger.provider}) not implemented."
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
            cfg.task.optimizer.target != "deepspeed.ops.adam.FusedAdam":
            warning_message(
                "PytorchLightning recommends FusedAdam optimizer "
                "when using DeepSpeed parallel backend "
                "(currently using '{}')".format(cfg.task.optimizer.target)
            )

    return cfg


def log_config(cfg: DictConfig):
    """Save configuration."""
    path = cfg.logger.runtime.artifact_path

    if not OmegaConf.is_config(cfg):
        error_message(
            "Attempting to save a config artifact but the object "
            "provided is not of type omegaconf.dictconfig.DictConfig.")
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    OmegaConf.save(cfg, f=os.path.join(path, "cfg.yaml"))