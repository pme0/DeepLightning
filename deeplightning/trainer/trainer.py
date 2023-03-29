from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
import os
from omegaconf import OmegaConf
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (GradientAccumulationScheduler,
                                         LearningRateMonitor,
                                         ModelCheckpoint,
                                         EarlyStopping)
from pytorch_lightning.loggers import WandbLogger

from deeplightning.config.defaults import __ConfigGroups__
from deeplightning.config.load import log_config
from deeplightning.logger.helpers import add_logger_params_to_config
from deeplightning.logger.wandb import init_wandb_metrics
from deeplightning.utils.messages import config_print
from deeplightning.utils.registry import (__LoggerRegistry__, 
                                          __HooksRegistry__)


class DLTrainer(Trainer):
    """ Deep Lightning Trainer.

    Inherits from `pytorch_lightning.Trainer`
    """
    def __init__(self, cfg: OmegaConf, args: dict) -> None:

        # Initialise logger and updated config with logger runtime parameters
        self.cfg, logger = self.init_logger(cfg)

        # Initialise callbacks
        callbacks = self.init_callbacks(cfg, logger.artifact_path)

        # Initialise parent class
        args = {
            **args, 
            "logger": logger, 
            "callbacks": callbacks
        }
        super().__init__(**args)


    def init_logger(self, cfg: OmegaConf) -> None:
        """ Initialize logger
        """

        #logger = __LoggerRegistry__[cfg.logger.name](cfg = cfg, logged_metric_names = __HooksRegistry__[cfg.task]["LOGGED_METRICS_NAMES"])
        if cfg.logger.name == "wandb":
            logger = WandbLogger(
                project = cfg.logger.project_name,
                notes = cfg.logger.notes,
                tags = cfg.logger.tags,
                log_model = "all",
            )
            logger.run_id = logger.experiment.id
            logger.run_name = logger.experiment.name
            logger.run_dir = logger.experiment.dir.replace("/files", "")
            logger.artifact_path = logger.experiment.dir

            # add logger params to config - so that it can be stored with the runtime parameters
            cfg = add_logger_params_to_config(
                cfg = cfg,
                run_id = logger.run_id,
                run_name = logger.run_name,
                run_dir = logger.run_dir,
                artifact_path = logger.artifact_path,
            )

            logger.step_label = init_wandb_metrics(
                metric_names = __HooksRegistry__[cfg.task]["LOGGED_METRICS_NAMES"], #["train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"], 
                step_label = "iteration",
            )

            log_config(cfg=cfg, path=logger.artifact_path)
            config_print(OmegaConf.to_yaml(cfg))


        # ensure all required attributes have been initialised
        attributes = ["run_id", "run_name", "run_dir", "artifact_path"]
        for attribute in attributes:
            if not hasattr(logger, attribute):
                raise AttributeError(f"Attribute '{attribute}' has not been set in DLLoger")
            
        return cfg, logger


    def init_callbacks(self, cfg: OmegaConf, artifact_path: str) -> List[Any]:
        """ Initialize callback functions
        """
        callbacks = []

        # ACCUMULATE GRADIENTS: scheduling={X: Y} means start 
        # accumulating from epoch X (0-indexed) and accumulate 
        # every Y batches
        accumulator = GradientAccumulationScheduler(
            scheduling={
                cfg.train.grad_accum_from_epoch: 
                cfg.train.grad_accum_every_n_batches}
        )
        callbacks += [accumulator]

        # TRACK LEARNING RATE: logged at the same frequency 
        # as `log_every_n_steps` in Trainer()
        lr_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        callbacks += [lr_monitor]

        # MODEL CHECKPOINTING: save model 'every_n_epochs'
        filename_metric = "" #TODO make this user-configurable OR set automatically from task
        checkpoint = ModelCheckpoint(
            dirpath = artifact_path,
            every_n_epochs = cfg.train.ckpt_every_n_epochs,
            save_last = False,
            save_top_k = cfg.train.ckpt_save_top_k,
            monitor = "val_acc",
            mode = "max",
            filename = "{epoch}-{step}-{val_acc:.4f}",  #TODO put filename_metric here
            save_on_train_epoch_end = False # False: save at validation_epoch_end
        )
        callbacks += [checkpoint]

        # EARLY STOPPING: stop training when 'monitor' metric asymptotes
        if cfg.train.early_stop_metric is not None:
            earlystopping = EarlyStopping(
                monitor = cfg.train.early_stop_metric,
                min_delta = cfg.train.early_stop_delta,
                patience = cfg.train.early_stop_patience,
                check_on_train_epoch_end = False # False: check at validation_epoch_end
            )
            callbacks += [earlystopping]

        return callbacks
