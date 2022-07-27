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
