from typing import Tuple
from omegaconf import OmegaConf
from torch import Tensor
import pytorch_lightning as pl
import wandb

from deeplightning.init.imports import init_obj_from_config
from deeplightning.utilities.messages import info_message
from deeplightning.trainer.metrics import metric_accuracy
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch
from deeplightning.logger.logwandb import initilise_wandb_metrics


class ImageClassification(pl.LightningModule):
    """ Task module for Image Classification. 

    LOGGING: manual logging `self.logger.log()` is used. This
    is more flexible as PyTorchLightning automatic logging 
    `self.log()`) only allows scalars, not histograms, images, etc.
    Additionally, auto-logging doesn't log at step 0, which is useful.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)
    
        self.sanity_check = True # to avoid logging sanity check metrics

        # optional metrics to use during training
        self.accuracy = metric_accuracy
        self.confusion_matrix = MetricsConfusionMatrix(cfg)
        
        # aggregation utilities
        self.gather_on_step = gather_on_step
        self.gather_on_epoch = gather_on_epoch

        # PyTorch-Lightning's model summary does not give the 
        # correct  number of trainable parameters; see 
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/12130
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info_message("Trainable parameters: {:,d}".format(self.trainable_params))
       
        # WandB logging:
        if self.cfg.logger.log_to_wandb:
            self.step_label = initilise_wandb_metrics(
                metrics = ["train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"], 
                step_label = "iteration",
            )


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
        if self.global_step % self.cfg.logger.log_every_n_steps == 0:

            # aggregate metrics across all devices
            metrics = self.gather_on_step(
                step_outputs = training_step_outputs, 
                metrics = ["loss", "train_acc"], 
                average = False)

            # chenge key from 'loss' to 'train_loss' (see `training_step()` for why)
            metrics['train_loss']  = metrics.pop('loss')

            # log training metrics
            if self.cfg.logger.log_to_wandb:
                metrics[self.step_label] = self.global_step
                wandb.log(metrics)
            else:
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
        if self.cfg.logger.log_to_wandb:
            metrics = {"train_acc": training_step_outputs[-1]["train_acc"].item()}
            metrics[self.step_label] = self.global_step
            wandb.log(metrics)
        else:
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
        preds = torch.argmax(logits, dim=1)
        
        # compute metrics
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)
        self.confusion_matrix.update(preds, y)
        
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
        
        # confusion matrix - TODO check that `torchmetrics.ConfusionMatrix()` gathers from multiple gpus
        cm = self.confusion_matrix.compute()
        cm_fig = self.confusion_matrix.draw(cm)
        metrics["val_confusion_matrix"] =  wandb.Image(cm_fig, caption="Confusion Matrix (%)")
        self.confusion_matrix.reset()

        # log validation metrics
        if self.cfg.logger.log_to_wandb:
            metrics[self.step_label] = self.global_step
            if not self.sanity_check:
                wandb.log(metrics)
            self.sanity_check = False
        else:
            self.logger.log_metrics(metrics, step = self.global_step)

        # EarlyStopping callback reads from `self.log()` but 
        # not from `self.logger.log()` thus this line. The key 
        # `m = self.cfg.train.early_stop_metric` must exist
        # in `validation_epoch_outputs`.
        if self.cfg.train.early_stop_metric is not None:
            m = self.cfg.train.early_stop_metric
            self.log(m, metrics[m])


    def test_step(self, batch, batch_idx):
        """ Hook for test step.
        """
        # forward pass
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        # compute metrics
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)
        self.confusion_matrix.update(preds, y)

        return {"test_loss": loss, 
                "test_acc": acc}


    def test_step_end(self, test_step_outputs):
        """ Hook for test step_end.

        Arguments:
            :test_step_outputs: (dict, list[dict]) metrics 
                dictionary in single-device training, or list of 
                metrics dictionaries in multi-device training (one 
                element per device).
                The output from `test_step()`.
        """

        # aggregate metrics across all devices.
        metrics = self.gather_on_step(
            step_outputs = test_step_outputs, 
            metrics = ["test_loss", "test_acc"], 
            average = False)

        return metrics


    def test_epoch_end(self, test_epoch_outputs):
        """ Hook for test epoch_end.

        Arguments:
            :test_epoch_outputs: (dict, list[dict]) metrics 
                dictionary in single-device training, or list of 
                metrics dictionaries in multi-device training (one 
                element per device). 
                The output from `test_step_end()`.
        """

        # aggregate losses across all steps and average
        metrics = self.gather_on_epoch(
            epoch_outputs = test_epoch_outputs, 
            metrics = ["test_loss", "test_acc"], 
            average = True)

        # confusion matrix
        cm = self.confusion_matrix.compute()
        cm_fig = self.confusion_matrix.draw(cm)
        metrics["test_confusion_matrix"] =  wandb.Image(cm_fig, caption="Confusion Matrix (%)")
        self.confusion_matrix.reset()
        
        # log test metrics
        if self.cfg.logger.log_to_wandb:
            metrics[self.step_label] = self.global_step
            wandb.log(metrics)
        else:
            self.logger.log_metrics(metrics, step = self.global_step)
        
