from typing import Tuple
from omegaconf import OmegaConf
from torch import Tensor
import pytorch_lightning as pl

from deeplightning.init.imports import init_obj_from_config
from deeplightning.utilities.messages import info_message
from deeplightning.config.helpers import field_exists_and_is_not_null
from deeplightning.trainer.metrics import metric_accuracy
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch


class ImageClassification(pl.LightningModule):
    """ Task module for Image Classification. 

    LOGGING: manual logging `self.logger.log()` is used. This
    is more flexible as PyTorchLightning automatic logging 
    `self.log()`) only allows scalars, not histograms, images, etc.
    Additionally, auto-logging doesn't log at step 0, which is useful.

    """

    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.loss = init_obj_from_config(config.model.loss)
        self.model = init_obj_from_config(config.model.network)
        self.optimizer = init_obj_from_config(config.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(config.model.scheduler, self.optimizer)
    
        # optional metrics to use during training
        self.accuracy = metric_accuracy

        # aggregation utilities
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
                "interval": self.config.model.scheduler.call.interval,
                "frequency": self.config.model.scheduler.call.frequency,
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
        """ Hook for training step.
        """
        # forward pass
        x, y = batch
        logits = self(x)

        # compute metrics
        train_loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)

        # `training_step()` expects one output with 
        # key 'loss'. This will be logged as 'train_loss' 
        # in `training_step_end()`.
        return {"loss": train_loss, 
                "train_acc": acc,}


    def training_step_end(self, training_step_outputs):
        """ Hook for training step_end.

        Arguments:
            :training_step_outputs: (dict, list[dict]) metrics 
                dictionary in single-device training, or list of 
                metrics dictionaries in multi-device training (one 
                element per device).
                The output from `training_step()`.
        """
        if self.global_step % self.config.logger.log_every_n_steps == 0:

            # aggregate metrics across all devices
            metrics = self.gather_on_step(
                step_outputs = training_step_outputs, 
                metrics = ["loss", "train_acc"], 
                average = False)

            # chenge key from 'loss' to 'train_loss' (see `training_step()` for why)
            metrics['train_loss']  = metrics.pop('loss')

            # log training metrics
            self.logger.log_metrics(
                metrics = metrics, 
                step = self.global_step)


    def training_epoch_end(self, training_step_outputs):
        """ Hook for training epoch_end.
        
        Arguments:
            :training_step_outputs: (dict, list[dict]) metrics 
                dictionary in single-device training, or list of 
                metrics dictionaries in multi-device training (one 
                element per device). 
                The output from `training_step()`.
        """

        # log training metrics on the last batch only
        self.logger.log_metrics(
            metrics = {
                "train_acc": training_step_outputs[-1]["train_acc"].item()}, 
            step = self.global_step)
    

    def validation_step(self, batch, batch_idx):
        """ Hook for validation step.
        """
        # forward pass
        x, y = batch
        logits = self(x)

        # compute metrics
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)

        return {"val_loss": loss, 
                "val_acc": acc}


    def validation_step_end(self, validation_step_outputs):
        """ Hook for validation step_end.

        Arguments:
            :validation_step_outputs: (dict, list[dict]) metrics 
                dictionary in single-device training, or list of 
                metrics dictionaries in multi-device training (one 
                element per device).
                The output from `validation_step()`.
        """

        # aggregate metrics across all devices.
        metrics = self.gather_on_step(
            step_outputs = validation_step_outputs, 
            metrics = ["val_loss", "val_acc"], 
            average = False)

        return metrics


    def validation_epoch_end(self, validation_epoch_outputs):
        """ Hook for validation epoch_end.

        Arguments:
            :validation_epoch_outputs: (dict, list[dict]) metrics 
                dictionary in single-device training, or list of 
                metrics dictionaries in multi-device training (one 
                element per device). 
                The output from `validation_step_end()`.
        """

        # aggregate losses across all steps and average
        metrics = self.gather_on_epoch(
            epoch_outputs = validation_epoch_outputs, 
            metrics = ["val_loss", "val_acc"], 
            average = True)

        # log validation metrics
        self.logger.log_metrics(metrics, step = self.global_step)

        # EarlyStopping callback reads from `self.log()` but 
        # not from `self.logger.log()` thus this line. The key 
        # `m = self.config.train.early_stop_metric` must exist
        # in `validation_epoch_outputs`.
        if field_exists_and_is_not_null(self.config.train, "early_stop_metric"):
            m = self.config.train.early_stop_metric
            self.log(m, metrics[m])