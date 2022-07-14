from omegaconf import OmegaConf
import os
import wandb

from deeplightning.logger import logging


class wandbLogger():
    """
    """
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        self.artifact_path = os.path.join(wandb.run.dir)
        logging.log_config(cfg, self.artifact_path)