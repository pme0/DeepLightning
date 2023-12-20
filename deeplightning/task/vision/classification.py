from typing import Any, Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision.utils import save_image
from lightning.pytorch.trainer.states import RunningStage

from deeplightning import TASK_REGISTRY
from deeplightning.init.imports import init_obj_from_config
from deeplightning.metrics.base import Metrics
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
            "test": ["classification_accuracy", "confusion_matrix", "precision_recall_curve"]}
        self.metrics = Metrics(cfg=cfg, defaults=self.default_task_metrics)

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
        #preds = torch.argmax(outputs, dim=1)

        # Update loss
        train_loss = self.loss(outputs, batch["targets"])
        self.training_step_outputs.append(train_loss)

        # Update metrics
        self.metrics.update(
            stage = "train", 
            **{
                "preds": outputs, 
                "target": batch["targets"],
            })
    
        if self.global_step % self.cfg.logger.log_every_n_steps == 0:
            
            metrics = self.init_metrics_dict(self.global_step)

            # Compute loss
            metrics["train_loss"] = torch.stack(self.training_step_outputs).mean()
            self.training_step_outputs.clear()
            
            # Compute metrics (batch only)
            self.metrics.compute(
                stage = "train",
                curr_metrics = metrics,
                reset=True,
                **{})
   
            # Logging
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
        #preds = torch.argmax(outputs, dim=1)

        # Update loss
        val_loss = self.loss(outputs, batch["targets"])
        self.validation_step_outputs.append(val_loss)

        # Update metrics
        self.metrics.update(
            stage = "val",
            **{
                "preds": outputs, 
                "target": batch["targets"],
            })


    def on_validation_epoch_end(self):

        metrics = self.init_metrics_dict(self.global_step)

        # Compute loss
        metrics["val_loss"] = torch.stack(self.validation_step_outputs).mean().item()
        self.validation_step_outputs.clear()

        # Compute metrics
        self.metrics.compute(
            stage = "val",
            curr_metrics = metrics,
            reset = True,
            **{
                "epoch": self.current_epoch,
                "max_epochs": self.trainer.max_epochs,
            })
       
        # Logging
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:  # `and self.global_step > 0`
            self.logger.log_metrics(metrics)

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

        # Convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # Compute forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)
        #preds = torch.argmax(outputs, dim=1)
                
        # Update loss
        test_loss = self.loss(outputs, batch["targets"])
        self.test_step_outputs.append(test_loss)

        # Update metrics
        self.metrics.update(
            stage = "test",
            **{
                "preds": outputs, 
                "target": batch["targets"],
            })


    def on_test_epoch_end(self):

        metrics = self.init_metrics_dict(self.global_step)

        # Compute loss
        metrics["test_loss"] = torch.stack(self.test_step_outputs).mean().item()
        self.test_step_outputs.clear()

        # Compute metrics
        self.metrics.compute(
            stage = "test",
            curr_metrics = metrics,
            reset = True,
            **{
                # `current_epoch` seems to be incremented after the last validation
                # loop so it's 1 more than it should be during the testing loop
                "epoch": self.current_epoch-1,
                "max_epochs": self.trainer.max_epochs,
            })

        # Logging
        self.logger.log_metrics(metrics)


@TASK_REGISTRY.register_element()
def image_classification(**kwargs: Any) -> ImageClassificationTask:
    return ImageClassificationTask(**kwargs)