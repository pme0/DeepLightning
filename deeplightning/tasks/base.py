from typing import Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
import lightning as pl

from deeplightning.utils.messages import info_message


class BaseTask(pl.LightningModule):
    """Base task module.

    Args:
        cfg: yaml configuration object

    Attributes:
        cfg: (OmegaConf) yaml configuration object.
        step_label: (str) label to track/log metrics against.
        metric_tracker: (dict) to store metrics.
        loss_tracker: (dict[dict]) to store losses.

    Notes:
        logging: Manual logging `self.logger.log()` is used. This is more 
            flexible as Lightning automatic logging `self.log()`) only 
            allows scalars, not histograms, images, etc. Additionally, 
            auto-logging doesn't log at step 0, which is useful.
        
        hooks: For *training*, the input to `training_epoch_end()` is the 
            set of outputs from `training_step()`. For *validation*, the 
            input to `validation_epoch_end()` is the output from 
            `validation_step_end()` and the input to `validation_step_end()` 
            is the output from `validation_step()`. See 
            <https://github.com/PyTorchLightning/pytorch-lightning/issues/9811>
        
        training_step outputs: If `return None`: "UserWarning: `training_step` 
            returned `None`. If this was on purpose, ignore this warning".
            If `return {}`: "MisconfigurationException: when `training_step` 
            returns a dict, the 'loss' key needs to be present".
    """
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg  #TODO check if this contains logger runtime params
        self.step_label = "iteration"

        # initialise metrics dictionary (to log)
        self.metric_tracker = {}
        # initialise loss accumulators (to store losses through steps)
        self.loss_tracker = {"train": {}, "val": {}, "test": {}}


    def on_task_init_end(self) -> None:
        """Additional attributes to initialise at the end of the `__init__` 
        method of the task class inheriting from `BaseTask` class.

        Attributes:
            num_trainable_params: mumber of trainable model parameters.
            num_nontrainable_params: mumber of nontrainable model parameters.
            num_total_params: mumber of total model parameters.
        """
        self.set_num_model_params()
        self.print_num_model_params()
 

    def on_logging_start(self) -> None:
        """Hook for start of logging.
        """
        # Set the appropriate label and step for the x-axis of the plots.
        self.metric_tracker[self.step_label] = self.global_step


    def on_logging_end(self, phase: str) -> None:
        """Hook for end of logging.
        """
        # Push metrics required by callbacls to default logger via `self.log()`
        if phase == "val":
            self.log_callback_metrics()
        # Push metrics to user defined logger via `self.logger.log()`
        self.logger.log_metrics(self.metric_tracker)
        # Reset metrics dictionary
        self.metric_tracker.clear()


    def log_callback_metrics(self):
        """Hook for logging metrics required for callbacks.

        Callbacks EarlyStopping and ModelCheckpoint read from `self.log()`, not
        from `self.logger.log()`, so we must log there. The following keys must
        exist in `metric_tracker`:
        > `m = self.cfg.stages.train.early_stop_metric` for EarlyStopping;
        > `m = self.cfg.stages.train.ckpt_monitor_metric` for ModelCheckpoint;
        """
        if self.cfg.stages.train.early_stop_metric is not None:
            m_earlystop = self.cfg.stages.train.early_stop_metric
            self.log(m_earlystop, self.metric_tracker[m_earlystop], sync_dist=True)
        if self.cfg.stages.train.ckpt_monitor_metric is not None:
            m_checkpoint = self.cfg.stages.train.ckpt_monitor_metric
            self.log(m_checkpoint, self.metric_tracker[m_checkpoint], sync_dist=True)


    def update_losses(self, phase: str, losses: dict) -> None:
        """Update loss accumulator with current step losses.

        Args:
            phase: trainer phase, either "train", "val", or "test".
            losses: current step losses.
        """
        for key in losses:
            if key not in self.loss_tracker[phase]:
                self.loss_tracker[phase][key] = []
            self.loss_tracker[phase][key].append(losses[key])
            

    def gather_losses(self, phase: str) -> None:
        """Aggregate losses across steps and distributed processes.

        Args:
            phase: trainer phase, either "train", "val", or "test".
        """
        for key in self.loss_tracker[phase]:
            # Aggregate across steps
            losses = torch.stack(self.loss_tracker[phase][key])
            # Aggregate across distributed processes
            losses = self.all_gather(losses)
            # Reduce
            losses = losses.mean().item()
            # Store
            self.metric_tracker[key] = losses
        # Reset accumulator
        self.loss_tracker[phase].clear()


    @property
    def num_trainable_params(self) -> int:
        return self._num_trainable_params
    

    @property
    def num_nontrainable_params(self) -> int:
        return self._num_nontrainable_params
    

    @property
    def num_total_params(self) -> int:
        return self._num_total_params


    def set_num_model_params(self) -> None:
        """Set the number of model parameters as attributes of the class.
        """
        self._num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        self._num_nontrainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad)
        self._num_total_params = self._num_trainable_params + self._num_nontrainable_params

    
    def print_num_model_params(self) -> None:
        """Print the number of model parameters.
        """
        info_message(
            f"No. trainable model parameters: {self.num_trainable_params:,d}")
        info_message(
            f"No. non-trainable model parameters: {self.num_nontrainable_params:,d}")
        info_message(
            f"No. total model parameters: {self.num_total_params:,d}")


    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


    def configure_optimizers(self) -> Tuple[dict]:
        raise NotImplementedError
    

    def training_step(self, batch, batch_idx):
        """ Lightning hook for training step.

        Args:
            batch: object containing the data output by the dataloader.
            batch_idx: index of batch
        """
        raise NotImplementedError


    def on_training_epoch_end(self):
        raise NotImplementedError
    

    def validation_step(self, batch, batch_idx):
        """ Lightning hook for validation step.

        Args:
            batch: object containing the data output by the dataloader.
            batch_idx: index of batch
        """
        raise NotImplementedError


    def on_validation_epoch_end(self):
        raise NotImplementedError


    def test_step(self, batch, batch_idx):
        """ Lightning hook for test step.

        Args:
            batch: object containing the data output by the dataloader.
            batch_idx: index of batch
        """
        raise NotImplementedError


    def on_test_epoch_end(self):
        raise NotImplementedError