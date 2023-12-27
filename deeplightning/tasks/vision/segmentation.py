from typing import Any, Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision.utils import save_image
from lightning.pytorch.trainer.states import RunningStage
import os

from deeplightning import TASK_REGISTRY
from deeplightning.utils.init.imports import init_obj_from_config
from deeplightning.metrics.base import Metrics
from deeplightning.tasks.base import BaseTask
from deeplightning.trainer.batch import dictionarify_batch


def process_model_outputs(outputs, model):
    """Processes model outouts and selects the appropriate elements
    """
    if model.__class__.__name__ == "DeepLabV3":
        # `DeepLabV3` returns a dictionaty with keys `out` (segmentation 
        # mask) and optionally `aux` if an auxiliary classifier is used.
        return outputs["out"]
    else:
        return outputs


class ImageSemanticSegmentationTask(BaseTask):
    """ Task module for Image Semantic Segmentation. 

    Args:
        cfg: yaml configuration object
    """
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg=cfg)
        
        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)
        
        self.default_metrics_dict = {
            "train": ["classification_accuracy"],
            "val": ["classification_accuracy", "confusion_matrix", "precision_recall_curve", "iou"],
            "test": ["classification_accuracy", "confusion_matrix", "precision_recall_curve", "iou"]}
        self.metrics = Metrics(cfg=cfg, defaults=self.default_metrics_dict)

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

        # Update losses
        train_loss = self.loss(outputs, batch["masks"])
        self.update_losses(phase="train", losses={"train_loss": train_loss})

        # Update metrics
        self.metrics.update(
            phase = "train", 
            **{
                "preds": outputs, 
                "target": batch["masks"],
            })

        if self.global_step % self.cfg.logger.log_every_n_steps == 0:

            self.on_logging_start()

            # Compute losses
            self.gather_losses(phase="train")

            # Compute metrics (batch only)
            self.metrics.compute(
                phase = "train",
                metric_tracker = self.metric_tracker,
                reset = True,
                **{})
           
            self.on_logging_end(phase="train")

        return {"loss": train_loss}  # see "training_step outputs" note in `BaseTask`
    

    def on_training_epoch_end(self):
        pass


    def validation_step(self, batch, batch_idx):

        # Convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)
            
        # Compute forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)
        
        preds = torch.argmax(outputs, dim=1)
        folder = "/Users/pme/Downloads/segm"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i in range(5):
            torch.save(obj=preds[i], f=os.path.join(folder, f"mask_step{self.global_step}_i{i}.pt"))
            save_image(
                tensor = preds[i].unsqueeze(0).float(), 
                fp = os.path.join(folder, f"{batch['masks_paths'][i].split('/')[-1]}_pred_step{self.global_step}.png")
            )

        # Update losses
        val_loss = self.loss(outputs, batch["masks"])
        self.update_losses(phase="val", losses={"val_loss": val_loss})

        # Update metrics
        self.metrics.update(
            phase = "val",
            **{
                "preds": outputs, 
                "target": batch["masks"],
                #"inputs_paths": batch["inputs_paths"],
                #"masks_paths": batch["masks_paths"],
            })
        

    def on_validation_epoch_end(self):

        self.on_logging_start()

        # Compute losses
        self.gather_losses(phase="val")

        # Compute metrics
        self.metrics.compute(
            phase = "val",
            metric_tracker = self.metric_tracker,
            reset = True,
            **{
                "epoch": self.current_epoch,
                "max_epochs": self.trainer.max_epochs,
            })

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            self.on_logging_end(phase="val")


    def test_step(self, batch, batch_idx):

        # Convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # Compute forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)
        #preds = torch.argmax(outputs, dim=1)
                
        # Update losses
        test_loss = self.loss(outputs, batch["masks"])
        self.update_losses(phase="test", losses={"test_loss": test_loss})

        # Update metrics
        self.metrics.update(
            phase = "test",
            **{
                "preds": outputs, 
                "target": batch["masks"],
            })


    def on_test_epoch_end(self):

        self.on_logging_start()

        # Compute loss
        self.gather_losses(phase="test")

        # Compute metrics
        self.metrics.compute(
            phase = "test",
            metric_tracker = self.metric_tracker,
            reset = True,
            **{
                "epoch": self.current_epoch-1,   # `current_epoch` seems to be incremented after the last validation loop so it's 1 more than it should be during the testing loop
                "max_epochs": self.trainer.max_epochs,
            })
        
        self.on_logging_end(phase="test")


@TASK_REGISTRY.register_element()
def image_semantic_segmentation(**kwargs: Any) -> ImageSemanticSegmentationTask:
    return ImageSemanticSegmentationTask(**kwargs)