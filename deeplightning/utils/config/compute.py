from typing import Any
from omegaconf import OmegaConf



def search_config(config: OmegaConf, value: Any = "AUTO", prepath=()):
    """Search for `value` in config keys and return the corresponding config paths
    """
    for k, v in config.items():
        path = prepath + (k,)
        if v == value: # found value
            yield path
        elif hasattr(v, 'items'): # v is a dict
            yield from search_config(v, value, path) 


def get_runtime_param(cfg: OmegaConf, cfg_path: str):
    """Compute runtime parameter
    """
    if cfg_path == "model.scheduler.args.T_max":
        target = "torch.optim.lr_scheduler.CosineAnnealingLR"
        assert OmegaConf.select(cfg, "model.scheduler.target") == target, f"invalid param `T_max` for scheduler `{target}`"
        num_batches = cfg.data.num_training_samples // cfg.data.batch_size + 1  #ceiling
        num_epochs = cfg.train.num_epochs
        return num_epochs * num_batches
    else:
        raise NotImplementedError


def runtime_compute(cfg: OmegaConf):
    """Update config parameters which require runtime computation
    """
    cfg_paths = list(search_config(cfg))
    for cfg_path in cfg_paths:
        dot_path = ".".join(cfg_path)
        value = get_runtime_param(cfg, dot_path)
        OmegaConf.update(cfg, dot_path, value)
    return cfg
