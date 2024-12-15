import time

from lightning import Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

from deeplightning.config.dlconfig import DeepLightningConfig
from deeplightning.utils.config.defaults import __ConfigGroups__
from deeplightning.utils.logger.helpers import add_logger_params_to_config
from deeplightning.utils.logger.wandb import init_wandb_metrics
from deeplightning.utils.python_utils import flatten_dict


class DeepLightningTrainer(Trainer):
    def __init__(self, cfg: DeepLightningConfig, args: dict) -> None:
        """ Deep Lightning Trainer."""

        self._passback_cfg, logger = self.init_logger(cfg)
        callbacks = self.init_callbacks(cfg, logger.artifact_path)
        super().__init__(**{
            **args, 
            "logger": logger, 
            "callbacks": callbacks,
            "deterministic": cfg.engine.seed is not None,
        })
        

    def init_logger(self, cfg: DeepLightningConfig) -> None:
        """ Initialize logger."""

        if cfg.logger.provider == "wandb":
            logger = WandbLogger(
                project = cfg.logger.project,
                notes = cfg.logger.notes,
                tags = cfg.logger.tags,
                log_model = "all",
            )
            
            # Sometimes `logger.experiment.dir` returns a function instead of the
            # expected string; this timeout loop attempts to fix this issue
            timeout = 0
            while not isinstance(logger.experiment.dir, str):
                if timeout > 5:
                    print(
                        "\n`logger.experiment.dir` is function instead of "
                        "string: {}\n\n".format(logger.experiment.dir)
                    )
                    raise AttributeError
                time.sleep(1)
                timeout += 1

            logger.run_id = logger.experiment.id
            logger.run_name = logger.experiment.name
            logger.artifact_path = logger.experiment.dir
            logger.run_dir = logger.experiment.dir.replace("/files", "")

            # Add logger params to config - so that it can be stored with the runtime parameters
            cfg = add_logger_params_to_config(
                cfg = cfg,
                run_id = logger.run_id,
                run_name = logger.run_name,
                run_dir = logger.run_dir,
                artifact_path = logger.artifact_path,
            )

            # Store config parameters - used in W&B for filtering experiments
            logger.experiment.config.update(flatten_dict(cfg))

            # Intialize step label for each metrics
            logger.step_label = init_wandb_metrics(
                metric_names = [f"{x}_{y}" for x in cfg.task.metrics for y in cfg.task.metrics[x]],
                step_label = "iteration",
            )

        else:
            raise NotImplementedError(
                f"Logger '{cfg.logger.provider}' not supported."
            )

        # Ensure all required attributes have been initialised
        attributes = [
            "run_id",
            "run_name",
            "run_dir",
            "artifact_path",
        ]
        for attribute in attributes:
            if not hasattr(logger, attribute):
                raise AttributeError(
                    f"Attribute '{attribute}' has not been set.")
            
        return cfg, logger


    def init_callbacks(self, cfg: DeepLightningConfig, artifact_path: str) -> list[Callback]:
        """ Initialize callback functions."""
        self.callbacks_dict = {}

        # ACCUMULATE GRADIENTS
        # scheduling={X: Y} means start accumulating from epoch X (0-indexed) 
        # and accumulate every Y batches
        accumulator = GradientAccumulationScheduler(
            scheduling={
                cfg.stages.train.grad_accum_from_epoch: 
                cfg.stages.train.grad_accum_every_n_batches}
        )
        self.callbacks_dict["accumulator"] = accumulator

        # TRACK LEARNING RATE
        # logged at the same frequency as `log_every_n_steps` in Trainer
        lr_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        self.callbacks_dict["lr_monitor"] = lr_monitor

        # MODEL CHECKPOINTING
        # save model `every_n_epochs`
        filename_metric = "" #TODO make this user-configurable OR set automatically from task
        checkpoint = ModelCheckpoint(
            dirpath = artifact_path,
            every_n_epochs = cfg.stages.train.ckpt_every_n_epochs,
            save_last = False,
            save_top_k = cfg.stages.train.ckpt_save_top_k,
            monitor = cfg.stages.train.ckpt_monitor_metric,
            mode = "max",
            filename = "{epoch}-{step}-{val_acc:.4f}",  #TODO put filename_metric here
            save_on_train_epoch_end = False # False: save at validation_epoch_end
        )
        self.callbacks_dict["checkpoint"] = checkpoint

        # EARLY STOPPING
        # stop training when 'monitor' metric asymptotes
        if cfg.stages.train.early_stop_metric is not None:
            earlystopping = EarlyStopping(
                monitor = cfg.stages.train.early_stop_metric,
                min_delta = cfg.stages.train.early_stop_delta,
                patience = cfg.stages.train.early_stop_patience,
                check_on_train_epoch_end = False # False: check at validation_epoch_end
            )
            self.callbacks_dict["earlystopping"] = earlystopping  

        # PROGRESS BAR
        # customize progress bar with:
        # ````
        # from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
        # RichProgressBar(
        #   theme=RichProgressBarTheme(
        #       description = "green_yellow",
        #       progress_bar = "green1",
        #       progress_bar_finished = "green1",
        #       progress_bar_pulse = "green1",
        #       batch_progress = "green_yellow",
        #       time = "grey82",
        #       processing_speed = "grey82",
        #       metrics = "grey82"))
        # ```
        progressbar = RichProgressBar()
        self.callbacks_dict["progressbar"] = progressbar      

        return list(self.callbacks_dict.values())  # Trainer takes a list
