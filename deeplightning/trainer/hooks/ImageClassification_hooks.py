from typing import Union, Any
import torch
import wandb

from deeplightning.trainer.batch import dictionarify_batch
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch


def training_step__ImageClassification(self, batch, batch_idx):
    """ Hook for `training_step`.

    Parameters
    ----------
    batch : object containing the data output by the dataloader.
    batch_idx : index of batch
    """

    # convert batch to dictionary form
    batch = dictionarify_batch(batch, self.cfg.data.dataset)

    # forward pass
    outputs = self.model(batch["inputs"])

    # loss
    train_loss = self.loss(outputs, batch["labels"])

    if "train_loss" not in self.training_step_outputs:
        self.training_step_outputs["train_loss"] = []
    self.training_step_outputs["train_loss"].append(train_loss)

    # metrics
    self.metrics["Accuracy_train"].update(preds = outputs, target = batch["labels"])

    # the output is not used but returning None gives the following warning
    # """lightning/pytorch/loops/optimization/automatic.py:129: 
    # UserWarning: `training_step` returned `None`. If this was 
    # on purpose, ignore this warning..."""
    return {"loss": train_loss}


def training_step_end__ImageClassification(self):
    """ Hook for `training_step_end`.
    """
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


def on_training_epoch_end__ImageClassification(self):
    """ Hook for `on_training_epoch_end`.
    """
    pass


def validation_step__ImageClassification(self, batch, batch_idx):
    """ Hook for `validation_step`.

    Parameters
    ----------
    batch : object containing the data output by the dataloader.
    batch_idx : index of batch
    """

    # convert batch to dictionary form
    batch = dictionarify_batch(batch, self.cfg.data.dataset)
        
    # forward pass
    outputs = self.model(batch["inputs"])
    preds = torch.argmax(outputs, dim=1)
        
    # loss
    val_loss = self.loss(outputs, batch["labels"])

    if "val_loss" not in self.validation_step_outputs:
        self.validation_step_outputs["val_loss"] = []
    self.validation_step_outputs["val_loss"].append(val_loss)

    # metrics
    self.metrics["Accuracy_val"].update(preds = preds, target = batch["labels"])
    self.metrics["ConfusionMatrix_val"].update(preds = preds, target = batch["labels"])
    self.metrics["PrecisionRecallCurve_val"].update(preds = outputs, target = batch["labels"])


def validation_step_end__ImageClassification(self):
    """ Hook for `validation_step_end`.
    """
    pass


def on_validation_epoch_end__ImageClassification(self):
    """ Hook for `on_validation_epoch_end`.
    """

    #TODO confirm on multi-gpu
    #print('\nself.validation_step_outputs["val_loss"]', len(self.validation_step_outputs["val_loss"]), '\n')

    metrics = {}
    metrics["val_loss"] = torch.stack(self.validation_step_outputs["val_loss"]).mean()
    self.validation_step_outputs.clear()  # free memory

    # accuracy
    metrics["val_acc"] = self.metrics["Accuracy_val"].compute()
    self.metrics["Accuracy_val"].reset()

    # confusion matrix
    cm = self.metrics["ConfusionMatrix_val"].compute()
    figure = self.metrics["ConfusionMatrix_val"].draw(
        confusion_matrix=cm, subset="val", epoch=self.current_epoch+1)
    metrics["val_confusion_matrix"] = wandb.Image(figure, 
        caption=f"Confusion Matrix [val, epoch {self.current_epoch+1}]")
    self.metrics["ConfusionMatrix_val"].reset()
        
    # precision-recall
    precision, recall, thresholds = self.metrics["PrecisionRecallCurve_val"].compute()
    figure = self.metrics["PrecisionRecallCurve_val"].draw(
        precision=precision, recall=recall, thresholds=thresholds, 
        subset="val", epoch=self.current_epoch+1)
    metrics["val_precision_recall"] = wandb.Image(figure, 
        caption=f"Precision-Recall Curve [val, epoch {self.current_epoch+1}]")
    self.metrics["PrecisionRecallCurve_val"].reset()

    # log validation metrics
    metrics[self.step_label] = self.global_step
    if not self.sanity_check:
        self.logger.log_metrics(metrics)
    self.sanity_check = False

    # The following is required for EarlyStopping and ModelCheckpoint callbacks to work properly. 
    # Callbacks read from `self.log()`, not from `self.logger.log()`, so need to log there.
    # [EarlyStopping] key `m = self.cfg.train.early_stop_metric` must exist in `metrics`
    if self.cfg.train.early_stop_metric is not None:
        m_earlystop = self.cfg.train.early_stop_metric
        self.log(m_earlystop, metrics[m_earlystop], sync_dist=True)
    # [ModelCheckpoint] key `m = self.cfg.train.ckpt_monitor_metric` must exist in `metrics`
    if self.cfg.train.ckpt_monitor_metric is not None:
        m_checkpoint = self.cfg.train.early_stop_metric
        if m_checkpoint != m_earlystop:
            self.log(m_checkpoint, metrics[m_checkpoint], sync_dist=True)


def test_step__ImageClassification(self, batch, batch_idx):
        """ Hook for `test_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader.
        batch_idx: index of batch.
        """

        # convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # forward pass
        outputs = self.model(batch["inputs"])
        preds = torch.argmax(outputs, dim=1)
            
        # loss
        test_loss = self.loss(outputs, batch["labels"])

        if "test_loss" not in self.test_step_outputs:
            self.test_step_outputs["test_loss"] = []
        self.test_step_outputs["test_loss"].append(test_loss)

        # metrics
        self.metrics["Accuracy_test"].update(preds = preds, target = batch["labels"])
        self.metrics["ConfusionMatrix_test"].update(preds = preds, target = batch["labels"])
        self.metrics["PrecisionRecallCurve_test"].update(preds = outputs, target = batch["labels"])
            

def test_step_end__ImageClassification(self):
    """ Hook for `test_step_end`.
    """
    pass


def on_test_epoch_end__ImageClassification(self):
    """ Hook for `on_test_epoch_end`.
    """

    metrics = {}
    metrics["test_loss"] = torch.stack(self.test_step_outputs["test_loss"]).mean()
    self.test_step_outputs.clear()  # free memory

    # accuracy
    metrics["test_acc"] = self.metrics["Accuracy_test"].compute()
    self.metrics["Accuracy_test"].reset()

    # confusion matrix
    cm = self.metrics["ConfusionMatrix_test"].compute()
    figure = self.metrics["ConfusionMatrix_test"].draw(
        confusion_matrix=cm, subset="test", epoch=self.current_epoch+1)
    metrics["test_confusion_matrix"] = wandb.Image(figure, 
        caption=f"Confusion Matrix [test, epoch {self.current_epoch+1}]")
    self.metrics["ConfusionMatrix_test"].reset()
        
    # precision-recall
    precision, recall, thresholds = self.metrics["PrecisionRecallCurve_test"].compute()
    figure = self.metrics["PrecisionRecallCurve_test"].draw(
        precision=precision, recall=recall, thresholds=thresholds, 
        subset="test", epoch=self.current_epoch+1)
    metrics["test_precision_recall"] = wandb.Image(figure, 
        caption=f"Precision-Recall Curve [test, epoch {self.current_epoch+1}]")
    self.metrics["PrecisionRecallCurve_test"].reset()

    # log test metrics
    metrics[self.step_label] = self.global_step
    self.logger.log_metrics(metrics)
