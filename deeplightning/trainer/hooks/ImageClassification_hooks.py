from typing import Union, Any
import torch
import wandb

from deeplightning.trainer.batch import dictionarify_batch
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch



def training_step__ImageClassification(self, batch, batch_idx):
    """ Hook for `training_step`.

    Parameters
    ----------
    batch : object containing the data output by the dataloader. For custom 
        datasets this is a dictionary with keys ["paths", "inputs", "labels"].
        For torchvision datasets, the function `dictionarify_batch()` is used
        to convert the native format to dictionary format
    batch_idx : index of batch
    """

    # convert batch to dictionary form
    batch = dictionarify_batch(batch, self.cfg.data.dataset)

    # forward pass
    outputs = self.model(batch["inputs"])

    # loss
    train_loss = self.loss(outputs, batch["labels"])

    # metrics
    self.metrics["Accuracy_train"].update(preds = outputs, target = batch["labels"])

    # `training_step()` expects one output with key 'loss'. 
    # This will be logged as 'train_loss' in `training_step_end()`.
    return {
        "loss": train_loss, 
    }


def training_step_end__ImageClassification(self, training_step_outputs):
    """ Hook for `training_step_end`.

    Parameters
    ----------
    training_step_outputs : (dict, list[dict]) metrics 
        dictionary in single-device training, or list of 
        metrics dictionaries in multi-device training (one 
        element per device). The output from `training_step()`.
        
    """
     
    if self.global_step % self.cfg.logger.log_every_n_steps == 0:

        # aggregate metrics across all devices
        metrics = gather_on_step(
            step_outputs = training_step_outputs, 
            metrics = ["loss"], 
            average = False)

        # chenge key from 'loss' to 'train_loss' (see `training_step()` for why)
        metrics['train_loss']  = metrics.pop('loss')

        # accuracy (batch only)
        metrics["train_acc"] = self.metrics["Accuracy_train"].compute()
        self.metrics["Accuracy_train"].reset()

        # log learning rate
        metrics['lr'] = self.lr_schedulers().get_last_lr()[0]
            
        # log training metrics
        metrics[self.step_label] = self.global_step
        self.logger_.log_metrics(metrics)


def training_epoch_end__ImageClassification(self, training_step_outputs):
    """ Hook for `training_epoch_end`.
        
    Parameters
    ----------
    training_step_outputs : (dict, list[dict]) metrics 
        dictionary in single-device training, or list of 
        metrics dictionaries in multi-device training (one 
        element per device). The output from `training_step()`.

    """

    # log training metrics on the last batch only
    #metrics = {"train_acc": training_step_outputs[-1]["train_acc"].item()}
    metrics = {}
    metrics[self.step_label] = self.global_step
    self.logger_.log_metrics(metrics)


def validation_step__ImageClassification(self, batch, batch_idx):
    """ Hook for `validation_step`.

    Parameters
    ----------
    batch : object containing the data output by the dataloader. For custom 
        datasets this is a dictionary with keys ["paths", "images", "labels"].
        For torchvision datasets, the function `dictionarify_batch()` is used
        to convert the native format to dictionary format
    batch_idx : index of batch

    """

    # convert batch to dictionary form
    batch = dictionarify_batch(batch, self.cfg.data.dataset)
        
    # forward pass
    outputs = self.model(batch["inputs"])
    preds = torch.argmax(outputs, dim=1)
        
    # loss
    loss = self.loss(outputs, batch["labels"])

    # metrics
    self.metrics["Accuracy_val"].update(preds = preds, target = batch["labels"])
    self.metrics["ConfusionMatrix_val"].update(preds = preds, target = batch["labels"])
    self.metrics["PrecisionRecallCurve_val"].update(preds = outputs, target = batch["labels"])
        
    return {
        "val_loss": loss, 
        #"val_acc": val_acc, #[TODO] needed for Early Stopping, see `validation_epoch_end` for more details
    }


def validation_step_end__ImageClassification(self, validation_step_outputs):
    """ Hook for `validation_step_end`.

    Parameters
    ----------
    validation_step_outputs : (dict, list[dict]) metrics 
        dictionary in single-device training, or list of 
        metrics dictionaries in multi-device training (one 
        element per device). The output from `validation_step()`.

    """

    # aggregate metrics across all devices.
    metrics = self.gather_on_step(
        step_outputs = validation_step_outputs, 
        metrics = ["val_loss"], 
        average = False)

    return metrics



def validation_epoch_end__ImageClassification(self, validation_epoch_outputs):
    """ Hook for `validation_epoch_end`.

    Parameters
    ----------
    validation_epoch_outputs : (dict, list[dict]) metrics 
        dictionary in single-device training, or list of 
        metrics dictionaries in multi-device training (one 
        element per device). 
        The output from `validation_step_end()`.

    """

    # aggregate losses across all steps and average
    metrics = self.gather_on_epoch(
        epoch_outputs = validation_epoch_outputs, 
        metrics = ["val_loss"], 
        average = True)

    # accuracy
    metrics["val_acc"] = self.metrics["Accuracy_val"].compute()
    self.metrics["Accuracy_val"].reset()

    # confusion matrix
    cm = self.metrics["ConfusionMatrix_val"].compute()
    figure = self.metrics["ConfusionMatrix_val"].draw(cm, subset="val", epoch=self.current_epoch+1)
    metrics["val_confusion_matrix"] = wandb.Image(figure, caption=f"Confusion Matrix [val, epoch {self.current_epoch+1}]")
    self.metrics["ConfusionMatrix_val"].reset()
        
    # precision-recall
    precision, recall, thresholds = self.metrics["PrecisionRecallCurve_val"].compute()
    figure = self.metrics["PrecisionRecallCurve_val"].draw(precision=precision, recall=recall, thresholds=thresholds, subset="val", epoch=self.current_epoch+1)
    metrics["val_precision_recall"] = wandb.Image(figure, caption=f"Precision-Recall Curve [val, epoch {self.current_epoch+1}]")
    self.metrics["PrecisionRecallCurve_val"].reset()

    # log validation metrics
    metrics[self.step_label] = self.global_step
    if not self.sanity_check:
        self.logger_.log_metrics(metrics)
    self.sanity_check = False

    # EarlyStopping callback reads from `self.log()` but not from `self.logger.log()` 
    # thus this line. The key `m = self.cfg.train.early_stop_metric` must exist
    # in `validation_epoch_outputs`.
    if self.cfg.train.early_stop_metric is not None:
        m = self.cfg.train.early_stop_metric
        self.log(m, metrics[m])


def test_step__ImageClassification(self, batch, batch_idx):
        """ Hook for `test_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        batch_idx: index of batch.
        
        """

        # convert batch to dictionary form
        batch = dictionarify_batch(batch, self.cfg.data.dataset)

        # forward pass
        outputs = self.model(batch["inputs"])
        preds = torch.argmax(outputs, dim=1)
            
        # loss
        loss = self.loss(outputs, batch["labels"])

        # metrics
        self.metrics["Accuracy_test"].update(preds = preds, target = batch["labels"])
        self.metrics["ConfusionMatrix_test"].update(preds = preds, target = batch["labels"])
        self.metrics["PrecisionRecallCurve_test"].update(preds = outputs, target = batch["labels"])
            
        return {"test_loss": loss}


def test_step_end__ImageClassification(self, test_step_outputs):
    """ Hook for `test_step_end`.

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
        metrics = ["test_loss"], 
        average = False)

    return metrics


def test_epoch_end__ImageClassification(self, test_epoch_outputs):
    """ Hook for `test_epoch_end`.

    Parameters
    ----------
    test_epoch_outputs : (dict, list[dict]) metrics 
        dictionary in single-device training, or list of 
        metrics dictionaries in multi-device training (one 
        element per device). 
        The output from `test_step_end()`.

    """

    # aggregate losses across all steps and average
    metrics = self.gather_on_epoch(
        epoch_outputs = test_epoch_outputs, 
        metrics = ["test_loss"], 
        average = True)

    # accuracy
    metrics["test_acc"] = self.metrics["Accuracy_test"].compute()
    self.metrics["Accuracy_test"].reset()

    # confusion matrix
    cm = self.metrics["ConfusionMatrix_test"].compute()
    figure = self.metrics["ConfusionMatrix_test"].draw(cm, subset="test", epoch=self.current_epoch+1)
    metrics["test_confusion_matrix"] = wandb.Image(figure, caption=f"Confusion Matrix [test, epoch {self.current_epoch+1}]")
    self.metrics["ConfusionMatrix_test"].reset()
        
    # precision-recall
    precision, recall, thresholds = self.metrics["PrecisionRecallCurve_test"].compute()
    figure = self.metrics["PrecisionRecallCurve_test"].draw(precision=precision, recall=recall, thresholds=thresholds, subset="test", epoch=self.current_epoch+1)
    metrics["test_precision_recall"] = wandb.Image(figure, caption=f"Precision-Recall Curve [test, epoch {self.current_epoch+1}]")
    self.metrics["PrecisionRecallCurve_test"].reset()

    # log test metrics
    if self.cfg.logger.log_to_wandb:
        metrics[self.step_label] = self.global_step
        self.logger_.log_metrics(metrics)
