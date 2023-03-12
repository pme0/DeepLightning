from typing import Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
import pytorch_lightning as pl
import wandb

from deeplightning.init.imports import init_obj_from_config
from deeplightning.init.initializers import init_metrics
from deeplightning.logger.logwandb import initilise_wandb_metrics
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch
from deeplightning.trainer.batch import dictionarify_batch
from deeplightning.trainer.hooks.ImageClassification_hooks import (
    training_step__ImageClassification,
    training_step_end__ImageClassification,
    training_epoch_end__ImageClassification,
    validation_step__ImageClassification,
    validation_step_end__ImageClassification,
    validation_epoch_end__ImageClassification,
    #test_step__ImageClassification,
    #test_step_end__ImageClassification,
    #test_epoch_end__ImageClassification,
)
from deeplightning.utils.messages import info_message
from deeplightning.utils.registry import __MetricsRegistry__



class ImageClassification(pl.LightningModule):
    """ Task module for Image Classification. 

    LOGGING: manual logging `self.logger.log()` is used. This
    is more flexible as PyTorchLightning automatic logging 
    `self.log()`) only allows scalars, not histograms, images, etc.
    Additionally, auto-logging doesn't log at step 0, which is useful.

    Parameters
    ----------
    cfg : yaml configuration object
    
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.model.network.params.num_classes
        self.classif_task = "binary" if self.num_classes == 2 else "multiclass"

        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)

        # PyTorch-Lightning performs a partial validation epoch to ensure that
        # everything is correct. Use this to avoid logging metrics to W&B for that 
        self.sanity_check = True

        # initialise metrics to track during training
        ImageClassification.metrics = init_metrics(cfg)

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
                metrics = ["train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "lr"], 
                step_label = "iteration",
            )

        # hook functions
        ImageClassification._training_step = training_step__ImageClassification
        ImageClassification._training_step_end = training_step_end__ImageClassification
        ImageClassification._training_epoch_end = training_epoch_end__ImageClassification
        ImageClassification._validation_step = validation_step__ImageClassification
        ImageClassification._validation_step_end = validation_step_end__ImageClassification
        ImageClassification._validation_epoch_end = validation_epoch_end__ImageClassification
        #ImageClassification._test_step = test_step__ImageClassification
        #ImageClassification._test_step_end = test_step_end__ImageClassification
        #ImageClassification._test_epoch_end = test_epoch_end__ImageClassification


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

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        batch_idx : index of batch

        """
        return self._training_step(batch, batch_idx)


    def training_step_end(self, training_step_outputs):
        """ Hook for training step_end.

        Parameters
        ----------
        training_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `training_step()`.

        """
        self._training_step_end(training_step_outputs)


    def training_epoch_end(self, training_step_outputs):
        """ Hook for training epoch_end.
        
        Parameters
        ----------
        training_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `training_step()`.

        """
        self._training_epoch_end(training_step_outputs)
    

    def validation_step(self, batch, batch_idx):
        """ Hook for validation step.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        batch_idx : index of batch

        """
        return self._validation_step(batch, batch_idx)


    def validation_step_end(self, validation_step_outputs):
        """ Hook for validation step_end.

        Parameters
        ----------
        validation_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `validation_step()`.

        """
        return self._validation_step_end(validation_step_outputs)


    def validation_epoch_end(self, validation_epoch_outputs):
        """ Hook for validation epoch_end.

        Parameters
        ----------
        validation_epoch_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). 
            The output from `validation_step_end()`.

        """
        self._validation_epoch_end(validation_epoch_outputs)


    def test_step(self, batch, batch_idx):
        """ Hook for test step.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        
        batch_idx: index of batch.
        
        """

        batch = dictionarify_batch(batch, self.dataset)

        # forward pass
        logits = self(batch["images"])
        preds = torch.argmax(logits, dim=1)
        
        # compute metrics
        loss = self.loss(logits, batch["labels"])
        acc = self.accuracy(logits=logits, target=batch["labels"], task=self.classif_task, num_classes=self.num_classes)
        self.confusion_matrix.update(preds=preds, target=batch["labels"])
        self.precision_recall.update(preds=logits, target=batch["labels"])

        return {"test_loss": loss, 
                "test_acc": acc}


    def test_step_end(self, test_step_outputs):
        """ Hook for test step_end.

        Parameters
        ----------
        test_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `test_step()`.

        """

        # aggregate metrics across all devices.
        metrics = self.gather_on_step(
            step_outputs = test_step_outputs, 
            metrics = ["test_loss", "test_acc"], 
            average = False)

        return metrics


    def test_epoch_end(self, test_epoch_outputs):
        """ Hook for test epoch_end.

        Parameters
        ----------
        test_epoch_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `test_step_end()`.
            
        """

        # aggregate losses across all steps and average
        metrics = self.gather_on_epoch(
            epoch_outputs = test_epoch_outputs, 
            metrics = ["test_loss", "test_acc"], 
            average = True)

        # confusion matrix
        cm = self.confusion_matrix.compute()
        figure = self.confusion_matrix.draw(cm, subset="test", epoch=self.current_epoch+1)
        metrics["test_confusion_matrix"] =  wandb.Image(figure, caption=f"Confusion Matrix [test]")
        self.confusion_matrix.reset()

        # precision-recall
        precision, recall, thresholds = self.precision_recall.compute()
        figure = self.precision_recall.draw(precision, recall, thresholds, subset="test", epoch=self.current_epoch+1)
        metrics["test_precision_recall"] = wandb.Image(figure, caption=f"Precision-Recall Curve [test, epoch {self.current_epoch+1}]")
        self.precision_recall.reset()

        # log test metrics
        if self.cfg.logger.log_to_wandb:
            metrics[self.step_label] = self.global_step
            wandb.log(metrics)
        else:
            self.logger.log_metrics(metrics, step = self.global_step)
        
