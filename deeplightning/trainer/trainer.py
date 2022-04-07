from typing import Any, Union, Tuple, Optional, List
ConfigElement = Union[str, int, float, None]
import os
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from deeplightning.config.defaults import __ConfigGroups__
from deeplightning.logger import logging
from deeplightning.init.imports import init_obj_from_config
from pytorch_lightning.callbacks import (GradientAccumulationScheduler,
                                         LearningRateMonitor,
                                         ModelCheckpoint,
                                         EarlyStopping)
from deeplightning.utilities.messages import info_message



class DLTrainer(Trainer):
    """ Deep Lightning Trainer.

    Inherits from `pytorch_lightning.Trainer`
    """
    def __init__(self, config: OmegaConf, args: dict) -> None:
        self.config = config
        logger = self.init_logger(config)
        callbacks = self.init_callbacks(config, logger.artifact_path)
        args = {
            **args, 
            "logger": logger,
            "callbacks": callbacks,
        }
        super().__init__(**args)


    def init_logger(self, config: OmegaConf) -> None:
        """ Initialize logger
        """
        logger = init_obj_from_config(config.logger)

        # BUG? without this line, the logger doesn't define 
        # `_experiment_id` and `_run_id`, which are needed
        # to create `artifact_path`
        logger.log_hyperparams(config) 

        # `artifact_path` is used to save model checkpoints, revised 
        # config file (after config checks/mods), and artifacts.
        # Add this path to the logger so it can be used anywhere.
        logger.artifact_path = os.path.join(
            logger._tracking_uri,
            logger._experiment_id,
            logger._run_id,
            "artifacts"
        )
        info_message("Artifact storage path: {}".format(logger.artifact_path))

        # add logging methods to the logger
        logger.log_config = logging.log_config
        logger.log_image = logging.log_image
        logger.log_figure = logging.log_figure
        logger.log_histogram = logging.log_histogram

        # log config file
        logger.log_config(config, logger.artifact_path)

        return logger


    def init_callbacks(self, config: OmegaConf, artifact_path: str) -> List[Any]:
        """ Initialize callback functions
        """
        callbacks = []

        # ACCUMULATE GRADIENTS: scheduling={X: Y} means start 
        # accumulating from epoch X (0-indexed) and accumulate 
        # every Y batches
        accumulator = GradientAccumulationScheduler(
            scheduling={
                config.train.grad_accum_from_epoch: 
                config.train.grad_accum_every_n_batches}
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
            every_n_epochs = config.train.ckpt_every_n_epochs,
            save_last = True,
        )
        callbacks += [checkpoint]

        # EARLY STOPPING: stop training when 'monitor' metric asymptotes
        if config.train.early_stop_metric is not None:
            earlystopping = EarlyStopping(
                monitor = config.train.early_stop_metric,
                min_delta = config.train.early_stop_delta,
                patience = config.train.early_stop_patience,
                check_on_train_epoch_end = False # False: check at validation_epoch_end
            )
            callbacks += [earlystopping]

        return callbacks
