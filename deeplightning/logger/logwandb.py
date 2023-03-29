from omegaconf import OmegaConf
import os
import wandb

from deeplightning.logger import logging


def get_artifact_path(logger_name: str) -> str:
    """
    """
    if logger_name == "wandb":
        return os.path.join(wandb.run.dir)
    else:
        raise ValueError(
            f"Unrecognized logger name '{logger_name}'. "
             "See `deeplightning.utils.registry.__LoggerRegistry__`.")
    

class Logger():
    """
    """
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        self.logger_name = cfg.logger.name
        self.artifact_path = get_artifact_path(self.logger_name)
        logging.log_config(cfg, self.artifact_path)


class wandbLogger():
    """
    """
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        self.artifact_path = os.path.join(wandb.run.dir)
        logging.log_config(cfg, self.artifact_path)
            

def initilise_wandb_metrics(metrics: list, step_label: str) -> None:
    """ Defines a custom x-axis metric `step_label` which is 
        synchronised with PyTprch-Lightning's `global_step`; and
        defines all other `metrics` to be plotted agains `step_label`
    """

    # define custom x-axis metric `step_label` (synchronised with PL `global_step`)
    wandb.define_metric(step_label)

    # initialise metrics to be plotted against `step_label`
    for m in metrics:
        wandb.define_metric(m, step_metric=step_label)

    return step_label
