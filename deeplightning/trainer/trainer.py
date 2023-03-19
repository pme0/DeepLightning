from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
import os
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (GradientAccumulationScheduler,
                                         LearningRateMonitor,
                                         ModelCheckpoint,
                                         EarlyStopping)

from deeplightning.utils.messages import info_message
from deeplightning.config.defaults import __ConfigGroups__
from deeplightning.logger import logging, logwandb
from deeplightning.init.imports import init_obj_from_config
from deeplightning.utils.registry import (__LoggerRegistry__, 
                                          __HooksRegistry__)



class DLTrainer(Trainer):
    """ Deep Lightning Trainer.

    Inherits from `pytorch_lightning.Trainer`
    """
    def __init__(self, cfg: OmegaConf, args: dict) -> None:

        # Logger
        # by default `pl.Trainer()` has an attribute `logger` so name 
        # this custom one `logger_` to avoid conflicts.
        #
        # [!] logger is being initialised twice: here (to get `artifact_path`) and 
        # in LightningModule to log TODO avoid this duplication
        artifact_path = self.init_logger(cfg).artifact_path

        # Pass callbacks to trainer
        callbacks = self.init_callbacks(cfg, artifact_path)
        args = {**args, "callbacks": callbacks}
        super().__init__(**args)


    def init_logger(self, cfg: OmegaConf) -> None:
        """ Initialize logger
        """

        # load logger
        logger = __LoggerRegistry__[cfg.logger.name](
            cfg = cfg,
            logged_metric_names = __HooksRegistry__[cfg.task]["LOGGED_METRICS_NAMES"]
        )

        # ensure all required attributes have been initialised
        attributes = ["run_id", "run_name", "run_dir", "artifact_path"]
        for attribute in attributes:
            if not hasattr(logger, attribute):
                raise AttributeError(f"Attribute '{attribute}' has not been set in DLLoger")
            
        return logger


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
        checkpoint = ModelCheckpoint(
            dirpath = artifact_path,
            every_n_epochs = cfg.train.ckpt_every_n_epochs,
            save_last = True,
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
