from omegaconf import DictConfig
from lightning import LightningModule, LightningDataModule

from deeplightning import TASK_REGISTRY
from deeplightning.core.trainer import DeepLightningTrainer
import deeplightning.tasks
from deeplightning.utils.config.process import log_config, resolve_config
from deeplightning.utils.imports import init_module
from deeplightning.utils.messages import print_config, info_message, warning_message


class DeepLightningPipeline():
    def __init__(self, cfg: DictConfig) -> None:

        # The following config is incomplete. After initializing the
        # Trainer, the config will be augmented with runtime parameters, 
        # e.g. run id which is created by the logger.
        cfg = resolve_config(cfg)

        # Initialize pipeline elements
        self.data = self._init_dataset(cfg)
        self.model = self._init_model(cfg)
        self.trainer = self._init_trainer(cfg)

        # Retrieve augmented config
        self.cfg = self.trainer.cfg
        
        # Log and Print augmented config
        log_config(self.cfg)
        print_config(self.cfg)

    def train(self) -> None:
        """Train model."""
        ckpt_path = self.cfg.train.ckpt_resume_path
        
        if ckpt_path is None:
            info_message("Starting training from scratch.")
        else:
            info_message(f"Resuming training from checkpoint '{ckpt_path}'.")
        
        self.trainer.fit(
            model = self.model,
            datamodule = self.data,
            ckpt_path = ckpt_path,
        )

    def eval(self, ckpt: str) -> None:
        """Evaluate model."""
        if ckpt == "best":
            self.eval_best()
        elif ckpt == "config":
            self.eval_ckpt()

    def eval_best(self) -> None:
        """Evaluate model using best checkpoint created during training."""
        info_message("Starting evaluation of best trained model.")
        self.trainer.test(
            model = self.model,
            ckpt_path = "best",
            datamodule = self.data,
        )

    def eval_ckpt(self) -> None:
        """Evaluate model using checkpoint specified in the config."""
        ckpt_path = self.cfg.test.ckpt_test_path
        info_message(f"Starting evaluation of checkpoint model '{ckpt_path}'.")
        self.trainer.test(
            model = self.model,
            ckpt_path = ckpt_path,
            datamodule = self.data,
        )

    def _init_dataset(self, cfg: DictConfig) -> LightningDataModule:
        """Initialize LightningDataModule.
        This contains data setup and loaders."""
        s = cfg.data.module
        return init_module(short_cfg = s, cfg = cfg)

    def _init_model(self, cfg: DictConfig) -> LightningModule:
        """Initialize LightningModule.
        This contains the task and training logic."""
        return TASK_REGISTRY.get_element_instance(
            name = cfg.task.name, 
            **{"cfg": cfg},
        )

    def _init_trainer(self, cfg: DictConfig) -> DeepLightningTrainer:
        """Initialize DeepLightning Trainer."""
        args = {
            "max_epochs": cfg.train.num_epochs,
            "num_nodes": cfg.engine.num_nodes,
            "accelerator": cfg.engine.accelerator,
            "strategy": cfg.engine.strategy,
            "devices": cfg.engine.devices,
            "precision": cfg.engine.precision,
            "check_val_every_n_epoch": cfg.train.val_every_n_epoch,
            "log_every_n_steps": cfg.logger.log_every_n_steps,
            }
        return DeepLightningTrainer(cfg, args)

    