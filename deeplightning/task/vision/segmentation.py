from typing import Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
import lightning as pl

from deeplightning.init.imports import init_obj_from_config
from deeplightning.init.initializers import init_metrics
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch
from deeplightning.utils.messages import info_message
from deeplightning.registry import __HooksRegistry__
from deeplightning.utils.metrics import classification_accuracy
from deeplightning.task.base import BaseTask


class SemanticSegmentationTask(BaseTask):
    """ Task module for Semantic Segmentation. 

    Args:
        cfg: yaml configuration object
    """
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg=cfg)
        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)
        
        # migration from `pytorch_lightning==1.5.10` to `lightning==2.0.0`
        self.training_step_outputs = {"train_loss": []}
        self.validation_step_outputs = {"val_loss": []}
        self.test_step_outputs = {"test_loss": []}

        # PyTorch-Lightning performs a partial validation epoch to ensure that
        # everything is correct. Use this to avoid logging metrics to W&B for that 
        self.sanity_check = True

        # Initialise metrics to track during training
        torch_device = torch.device("cuda") if cfg.engine.accelerator == "gpu" else torch.device('cpu')

        #self.metrics = init_metrics(cfg, device=torch_device)
        self.metrics = {
            "Accuracy_train": classification_accuracy(cfg),
            "Accuracy_val": classification_accuracy(cfg),
            "Accuracy_test": classification_accuracy(cfg),
        }

        # Initialise label to track metrics against
        self.step_label = "iteration"

        # Define hook functions
        # to make the hooks bound to the class (so that they can access class attributes 
        #  using `self.something`), the assignment must specify the class name as follows:
        # `ClassName.fn = my_fn` rather than `self.fn = my_fn`
        SemanticSegmentationTask._training_step = __HooksRegistry__[cfg.task]["training_step"]
        SemanticSegmentationTask._training_step_end = __HooksRegistry__[cfg.task]["training_step_end"]
        SemanticSegmentationTask._on_training_epoch_end = __HooksRegistry__[cfg.task]["on_training_epoch_end"]
        SemanticSegmentationTask._validation_step = __HooksRegistry__[cfg.task]["validation_step"]
        SemanticSegmentationTask._validation_step_end = __HooksRegistry__[cfg.task]["validation_step_end"]
        SemanticSegmentationTask._on_validation_epoch_end = __HooksRegistry__[cfg.task]["on_validation_epoch_end"]
        SemanticSegmentationTask._test_step = __HooksRegistry__[cfg.task]["test_step"]
        SemanticSegmentationTask._test_step_end = __HooksRegistry__[cfg.task]["test_step_end"]
        SemanticSegmentationTask._on_test_epoch_end = __HooksRegistry__[cfg.task]["on_test_epoch_end"]

        # Aggregation utilities
        self.gather_on_step = gather_on_step
        self.gather_on_epoch = gather_on_epoch

        # PyTorch-Lightning's model summary does not give the 
        # correct  number of trainable parameters; see 
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/12130
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info_message("Trainable parameters: {:,d}".format(self.trainable_params))
       

    def forward(self, x: Tensor) -> Tensor:
        """ Model forward pass.
        """
        return self.model(x)


    def configure_optimizers(self) -> Tuple[dict]:
        """ Configure optimizers and schedulers.
        """
        return ({
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": self.cfg.model.scheduler.call.interval,
                "frequency": self.cfg.model.scheduler.call.frequency,
            },
        })

    
    """ NOTE on training/validation hooks.

        For *training*, the input to `training_epoch_end()` is 
        the set of outputs from `training_step()`. For 
        *validation*, the input to `validation_epoch_end()` 
        is the output from `validation_step_end()` and the input 
        to `validation_step_end()` is the output from 
        `validation_step()`.

        https://github.com/PyTorchLightning/pytorch-lightning/issues/9811
    """


    def training_step(self, batch, batch_idx):
        """ Hook for `training_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader.
        batch_idx : index of batch
        """
        return self._training_step(batch, batch_idx)


    def training_step_end(self):
        """ Hook for `training_step_end`.
        """
        self._training_step_end()


    def on_training_epoch_end(self):
        """ Hook for `on_training_epoch_end`.
        """
        self._on_training_epoch_end()
    

    def validation_step(self, batch, batch_idx):
        """ Hook for `validation_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader.
        batch_idx : index of batch.

        """
        return self._validation_step(batch, batch_idx)


    def validation_step_end(self):
        """ Hook for `validation_step_end`.
        """
        return self._validation_step_end()


    def on_validation_epoch_end(self):
        """ Hook for `validation_epoch_end`.
        """
        self._on_validation_epoch_end()


    def test_step(self, batch, batch_idx):
        """ Hook for `test_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. 
        batch_idx: index of batch.
        """
        return self._test_step(batch, batch_idx)


    def test_step_end(self):
        """ Hook for `test_step_end`.
        """
        return self._test_step_end()


    def on_test_epoch_end(self):
        """ Hook for `on_test_epoch_end`.
        """
        self._on_test_epoch_end()