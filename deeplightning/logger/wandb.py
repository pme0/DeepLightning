from omegaconf import OmegaConf
import wandb

from deeplightning.logger.helpers import add_logger_params_to_config


class wandbLogger():
    """ Deep Lightning Logger.
    """
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        self.init_wandb()

    def init_wandb(self):
        wandb.init(
            project = self.cfg.logger.project_name,
            notes = self.cfg.logger.notes,
            tags = self.cfg.logger.tags,
        )

        # get logger runtime parameters
        self.run_id = wandb.run.id
        self.run_name = wandb.run.name
        self.run_dir = wandb.run.dir.replace("/files", "")
        self.artifact_path = wandb.run.dir

        # add logger params to config - so that it can be stored with the runtime parameters
        self.cfg = add_logger_params_to_config(self.cfg)


    def log_image(self):
        pass