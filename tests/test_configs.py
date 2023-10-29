import os
import sys
sys.path.insert(0, "..")
from omegaconf import OmegaConf
import pytest

from deeplightning.config.load import load_config


def test_configs():

    cfg_base = load_config(config_file="configs/_base.yaml")
    assert OmegaConf.is_config(cfg_base)

    for cfg_filename in os.listdir("configs"):
        if cfg_filename != "_base.yaml":
            cfg = load_config(config_file = f"configs/{cfg_filename}")
            assert OmegaConf.is_config(cfg)

   