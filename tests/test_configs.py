import os
from glob import glob
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

from deeplightning.config.load import load_config


def check_all_keys_exist(cfg_base, cfg):
    """Check if all keys in `cfg_base` exist in `cfg`
    """
    if not isinstance(cfg_base, DictConfig):
        return False
    if not isinstance(cfg, DictConfig):
        return False
    for key in cfg_base.keys():
        if isinstance(cfg_base[key], DictConfig):
            check_all_keys_exist(cfg_base[key], cfg[key])
        else:
            if key not in cfg:
                return False
    return True


def test_configs():

    cfg_base = load_config(config_file="configs/_base.yaml")
    assert OmegaConf.is_config(cfg_base)

    configs = glob(os.path.join("configs", "*")) + glob(os.path.join("tests/helpers", "*"))
    configs = [x for x in configs if x.endswith(".yaml") and not x.endswith("_base.yaml")]

    for cfg_fp in configs:
        cfg = load_config(config_file=cfg_fp)
        assert OmegaConf.is_config(cfg)
        assert check_all_keys_exist(cfg_base, cfg)

   