from typing import Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision.utils import save_image

import lightning as pl
from lightning.pytorch.trainer.states import RunningStage

from deeplightning.init.imports import init_obj_from_config
#from deeplightning.init.initializers import init_metrics
#from deeplightning.trainer.gather import gather_on_step, gather_on_epoch
from deeplightning.utils.messages import info_message
from deeplightning.registry import __HooksRegistry__
from deeplightning.utils.metrics import classification_accuracy
from deeplightning.task.base import BaseTask
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
        #self.sanity_check = True

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
        #SemanticSegmentationTask._training_step = __HooksRegistry__[cfg.task]["training_step"]
        #SemanticSegmentationTask._training_step_end = __HooksRegistry__[cfg.task]["training_step_end"]
        #SemanticSegmentationTask._on_training_epoch_end = __HooksRegistry__[cfg.task]["on_training_epoch_end"]
        #SemanticSegmentationTask._validation_step = __HooksRegistry__[cfg.task]["validation_step"]
        #SemanticSegmentationTask._validation_step_end = __HooksRegistry__[cfg.task]["validation_step_end"]
        #SemanticSegmentationTask._on_validation_epoch_end = __HooksRegistry__[cfg.task]["on_validation_epoch_end"]
        #SemanticSegmentationTask._test_step = __HooksRegistry__[cfg.task]["test_step"]
        #SemanticSegmentationTask._test_step_end = __HooksRegistry__[cfg.task]["test_step_end"]
        #SemanticSegmentationTask._on_test_epoch_end = __HooksRegistry__[cfg.task]["on_test_epoch_end"]

        # Aggregation utilities
        #self.gather_on_step = gather_on_step
        #self.gather_on_epoch = gather_on_epoch

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

        # convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)

        # loss
        train_loss = self.loss(outputs, batch["masks"])

        if "train_loss" not in self.training_step_outputs:
            self.training_step_outputs["train_loss"] = []
        self.training_step_outputs["train_loss"].append(train_loss)

        # metrics
        self.metrics["Accuracy_train"].update(preds=outputs, target=batch["masks"])

        if self.global_step % self.cfg.logger.log_every_n_steps == 0:

            metrics = {}
            metrics["train_loss"] = torch.stack(self.training_step_outputs["train_loss"]).mean()
            self.training_step_outputs.clear()  # free memory
            # accuracy (batch only)
            metrics["train_acc"] = self.metrics["Accuracy_train"].compute()
            self.metrics["Accuracy_train"].reset()
            # log learning rate
            #metrics['lr'] = self.lr_schedulers().get_last_lr()[0]

            # log training metrics
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

        # convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)
            
        # forward pass
        outputs = self.model(batch["inputs"])
        outputs = process_model_outputs(outputs, self.model)
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(5):
            print(batch["inputs_paths"][i])
            print(batch["masks_paths"][i])
            torch.save(obj=preds[i], f=f"/Users/pme/Downloads/segm/mask_step{self.global_step}_i{i}.pt")
            save_image(preds[i].unsqueeze(0).float(), fp=f"/Users/pme/Downloads/segm/mask_step{self.global_step}_i{i}.jpeg")

        # loss
        val_loss = self.loss(outputs, batch["masks"])

        if "val_loss" not in self.validation_step_outputs:
            self.validation_step_outputs["val_loss"] = []
        self.validation_step_outputs["val_loss"].append(val_loss)

        # metrics
        self.metrics["Accuracy_val"].update(preds = preds, target = batch["masks"])
        #self.metrics["ConfusionMatrix_val"].update(preds = preds, target = batch["masks"])
        #self.metrics["PrecisionRecallCurve_val"].update(preds = outputs, target = batch["masks"])


    def on_validation_epoch_end(self):

        metrics = {}
        metrics["val_loss"] = torch.stack(self.validation_step_outputs["val_loss"]).mean()
        self.validation_step_outputs.clear()  # free memory

        # accuracy
        metrics["val_acc"] = self.metrics["Accuracy_val"].compute()
        self.metrics["Accuracy_val"].reset()

        # confusion matrix
        '''
        cm = self.metrics["ConfusionMatrix_val"].compute()
        figure = self.metrics["ConfusionMatrix_val"].draw(
            confusion_matrix=cm, subset="val", epoch=self.current_epoch+1)
        metrics["val_confusion_matrix"] = wandb.Image(figure, 
            caption=f"Confusion Matrix [val, epoch {self.current_epoch+1}]")
        self.metrics["ConfusionMatrix_val"].reset()
        '''

        # precision-recall
        '''
        precision, recall, thresholds = self.metrics["PrecisionRecallCurve_val"].compute()
        figure = self.metrics["PrecisionRecallCurve_val"].draw(
            precision=precision, recall=recall, thresholds=thresholds, 
            subset="val", epoch=self.current_epoch+1)
        metrics["val_precision_recall"] = wandb.Image(figure, 
            caption=f"Precision-Recall Curve [val, epoch {self.current_epoch+1}]")
        self.metrics["PrecisionRecallCurve_val"].reset()
        '''

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
        test_loss = self.loss(outputs, batch["masks"])

        if "test_loss" not in self.test_step_outputs:
            self.test_step_outputs["test_loss"] = []
        self.test_step_outputs["test_loss"].append(test_loss)

        # metrics
        self.metrics["Accuracy_test"].update(preds = preds, target = batch["masks"])
            #self.metrics["ConfusionMatrix_test"].update(preds = preds, target = batch["masks"])
        #self.metrics["PrecisionRecallCurve_test"].update(preds = outputs, target = batch["masks"])


    def on_test_epoch_end(self):

        metrics = {}
        metrics["test_loss"] = torch.stack(self.test_step_outputs["test_loss"]).mean()
        self.test_step_outputs.clear()  # free memory

        # accuracy
        metrics["test_acc"] = self.metrics["Accuracy_test"].compute()
        self.metrics["Accuracy_test"].reset()

        # confusion matrix
        '''
        cm = self.metrics["ConfusionMatrix_test"].compute()
        figure = self.metrics["ConfusionMatrix_test"].draw(
            confusion_matrix=cm, subset="test", epoch=self.current_epoch+1)
        metrics["test_confusion_matrix"] = wandb.Image(figure, 
            caption=f"Confusion Matrix [test, epoch {self.current_epoch+1}]")
        self.metrics["ConfusionMatrix_test"].reset()    
        '''

        # precision-recall
        '''
        precision, recall, thresholds = self.metrics["PrecisionRecallCurve_test"].compute()
        figure = self.metrics["PrecisionRecallCurve_test"].draw(
            precision=precision, recall=recall, thresholds=thresholds, 
            subset="test", epoch=self.current_epoch+1)
        metrics["test_precision_recall"] = wandb.Image(figure, 
            caption=f"Precision-Recall Curve [test, epoch {self.current_epoch+1}]")
        self.metrics["PrecisionRecallCurve_test"].reset()
        '''

        # log test metrics
        metrics[self.step_label] = self.global_step
        self.logger.log_metrics(metrics)

