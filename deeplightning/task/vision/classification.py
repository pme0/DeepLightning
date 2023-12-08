from typing import Any, Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision.utils import save_image
from lightning.pytorch.trainer.states import RunningStage
import wandb

from deeplightning import TASK_REGISTRY
from deeplightning.init.imports import init_obj_from_config
from deeplightning.metrics.base import Metrics
from deeplightning.metrics.classification.accuracy import classification_accuracy
from deeplightning.metrics.classification.confusion_matrix import confusion_matrix
from deeplightning.metrics.classification.precision_recall import precision_recall_curve
from deeplightning.task.base import BaseTask
from deeplightning.trainer.batch import dictionarify_batch


def process_model_outputs(outputs, model):
    """Processes model outouts and selects the appropriate elements
    """
    if model.__class__.__name__ == "someModel":
        return outputs["someThing"]
    else:
        return outputs


class ImageClassificationTask(BaseTask):
    """ Task module for Image Classification. 

    Args:
        cfg: yaml configuration object
    """
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg=cfg)
        
        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)
        
        self.default_task_metrics = {
            "train": ["classification_accuracy"],
            "val": ["classification_accuracy", "confusion_matrix", "precision_recall_curve"],
            "test": ["classification_accuracy", "confusion_matrix", "precision_recall_curve"],}
        self.metrics = Metrics(cfg=cfg, defaults=self.default_task_metrics)
       # self.metrics = Metrics(cfg=cfg, defaults=self.default_task_metrics).md

        self.on_task_init_end()


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


    def training_step(self, batch, batch_idx):

        # Convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # Compute forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)

        # Compute loss
        train_loss = self.loss(outputs, batch["targets"])
        self.training_step_outputs.append(train_loss)

        # Compute metrics
        update_args = {"preds": outputs, "target": batch["targets"]}
        self.metrics.update(stage="train", **update_args)
    
        if self.global_step % self.cfg.logger.log_every_n_steps == 0:

            metrics = {} #self.on_logging_start(self.global_step)

            # Loss
            metrics["train_loss"] = torch.stack(self.training_step_outputs).mean()
            self.training_step_outputs.clear()  # free memory
            
            # Metrics (batch only)
            self.metrics.compute(existing_metrics=metrics, stage="train", reset=True)
   
            # Logging
            metrics[self.step_label] = self.global_step
            self.logger.log_metrics(metrics)

        # the output is not used but returning None gives the following warning
        # """lightning/pytorch/loops/optimization/automatic.py:129: 
        # UserWarning: `training_step` returned `None`. If this was 
        # on purpose, ignore this warning..."""
        return {"loss": train_loss}
    

    def on_training_epoch_end(self):
        pass


    def validation_step(self, batch, batch_idx):

        # Convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)
            
        # Compute forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)
        preds = torch.argmax(outputs, dim=1)

        # Compute forward loss
        val_loss = self.loss(outputs, batch["targets"])
        self.validation_step_outputs.append(val_loss)

        # Compute metrics
        update_args = {"preds": outputs, "target": batch["targets"]}
        self.metrics.update(stage="val", **update_args)
        #self.metrics.md["val"]["classification_accuracy"].update(preds=preds, target=batch["targets"])
        #self.metrics.md["val"]["confusion_matrix"].update(preds=preds, target=batch["targets"])
        #self.metrics.md["val"]["precision_recall_curve"].update(preds=outputs, target=batch["targets"])


    def on_validation_epoch_end(self):

        metrics = {}

        metrics["val_loss"] = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()  # free memory

        # accuracy
        self.metrics.compute(existing_metrics=metrics, stage="val", reset=True)
        #metrics["val_acc"] = self.metrics.md["val"]["classification_accuracy"].compute()
        #self.metrics.md["val"]["classification_accuracy"].reset()
        print(self.global_step, metrics["val_accuracy"])
        print(self.global_step, metrics["val_confusion_matrix"])
        raise

        # confusion matrix
        cm = self.metrics.md["val"]["confusion_matrix"].compute()
        figure = self.metrics.md["val"]["confusion_matrix"].draw(
            confusion_matrix = cm, 
            stage = "val", 
            epoch = self.current_epoch, 
            max_epochs = self.trainer.max_epochs,
        )
        caption = f"Confusion Matrix [val, epoch {self.current_epoch+1}/{self.trainer.max_epochs}]"
        metrics["val_confusion_matrix"] = wandb.Image(figure, caption=caption)
        self.metrics.md["val"]["confusion_matrix"].reset()

        # precision-recall
        precision, recall, thresholds = self.metrics.md["val"]["precision_recall_curve"].compute()
        figure = self.metrics.md["val"]["precision_recall_curve"].draw(
            precision = precision,
            recall = recall,
            thresholds = thresholds, 
            stage = "val", 
            epoch = self.current_epoch,
            max_epochs = self.trainer.max_epochs,
        )
        caption = f"Precision-Recall Curve [val, epoch {self.current_epoch+1}/{self.trainer.max_epochs}]"
        metrics["val_precision_recall"] = wandb.Image(figure, caption=caption)
        self.metrics.md["val"]["precision_recall_curve"].reset()

        # log validation metrics
        metrics[self.step_label] = self.global_step
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:  # `and self.global_step > 0`
            self.logger.log_metrics(metrics)
        #self.sanity_check = False

        # The following is required for EarlyStopping and ModelCheckpoint callbacks to work properly. 
        # Callbacks read from `self.log()`, not from `self.logger.log()`, so need to log there.
        # [EarlyStopping] key `m = self.cfg.train.early_stop_metric` must exist in `metrics`
        if self.cfg.train.early_stop_metric is not None:
            m_earlystop = self.cfg.train.early_stop_metric
            self.log(m_earlystop, metrics[m_earlystop], sync_dist=True)
        # [ModelCheckpoint] key `m = self.cfg.train.ckpt_monitor_metric` must exist in `metrics`
        if self.cfg.train.ckpt_monitor_metric is not None:
            m_checkpoint = self.cfg.train.ckpt_monitor_metric
            self.log(m_checkpoint, metrics[m_checkpoint], sync_dist=True)


    def test_step(self, batch, batch_idx):

        # convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)
        preds = torch.argmax(outputs, dim=1)
                
        # loss
        test_loss = self.loss(outputs, batch["targets"])
        self.test_step_outputs.append(test_loss)

        # metrics
        self.metrics.md["test"]["classification_accuracy"].update(preds=preds, target=batch["targets"])
        self.metrics.md["test"]["confusion_matrix"].update(preds=preds, target=batch["targets"])
        self.metrics.md["test"]["precision_recall_curve"].update(preds=outputs, target=batch["targets"])


    def on_test_epoch_end(self):

        metrics = {}
        metrics["test_loss"] = torch.stack(self.test_step_outputs).mean()
        self.test_step_outputs.clear()  # free memory

        # accuracy
        metrics["test_acc"] = self.metrics.md["test"]["classification_accuracy"].compute()
        self.metrics.md["test"]["classification_accuracy"].reset()

        # confusion matrix
        cm = self.metrics.md["test"]["confusion_matrix"].compute()
        figure = self.metrics.md["test"]["confusion_matrix"].draw(
            confusion_matrix = cm,
            stage = "test", 
            epoch = self.current_epoch,
            max_epochs = self.trainer.max_epochs,
        )
        caption = f"Confusion Matrix [test, epoch {self.current_epoch+1}/{self.trainer.max_epochs}]"
        metrics["test_confusion_matrix"] = wandb.Image(figure, caption=caption)
        self.metrics.md["test"]["confusion_matrix"].reset()    

        # precision-recall
        precision, recall, thresholds = self.metrics.md["test"]["precision_recall_curve"].compute()
        figure = self.metrics.md["test"]["precision_recall_curve"].draw(
            precision = precision, 
            recall = recall, 
            thresholds = thresholds, 
            stage = "test", 
            epoch = self.current_epoch,
            max_epochs = self.trainer.max_epochs,
        )
        caption = f"Precision-Recall Curve [test, epoch {self.current_epoch+1}/{self.trainer.max_epochs}]"
        metrics["test_precision_recall"] = wandb.Image(figure, caption=caption)
        self.metrics.md["test"]["precision_recall_curve"].reset()

        # log test metrics
        metrics[self.step_label] = self.global_step
        self.logger.log_metrics(metrics)


@TASK_REGISTRY.register_element()
def image_classification(**kwargs: Any) -> ImageClassificationTask:
    return ImageClassificationTask(**kwargs)