from typing import Tuple, Union, List
from omegaconf import OmegaConf
import torch
from torch import Tensor
import lightning as pl

from deeplightning.utils.messages import info_message



class BaseTask(pl.LightningModule):
    """Base task module.

    Note on logging:
        manual logging `self.logger.log()` is used. This is more 
        flexible as Lightning automatic logging `self.log()`) only 
        allows scalars, not histograms, images, etc./ Additionally, 
        auto-logging doesn't log at step 0, which is useful.
        
    Note on hooks: 
        For *training*, the input to `training_epoch_end()` is the 
        set of outputs from `training_step()`. For *validation*, the 
        input to `validation_epoch_end()` is the output from 
        `validation_step_end()` and the input to `validation_step_end()` 
        is the output from `validation_step()`. See 
        https://github.com/PyTorchLightning/pytorch-lightning/issues/9811

    Args:
        cfg: yaml configuration object

    Attributes:
        cfg: (OmegaConf) yaml configuration object.
        step_label: (str) label to track/log metrics against.
        metrics_logged: (dict) to store metrics.
        batch_losses: (dict[dict]) to store losses.
    """
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg  #TODO check if this contains logger runtime params
        self.step_label = "iteration"

        # initialise metrics dictionary (to log)
        self.metrics_logged = {}
        # initialise loss accumulators (to store losses through steps)
        self.batch_losses = {
            "train": {},
            "val": {},
            "test": {},
        }


    def on_task_init_end(self) -> None:
        """Additional attributes to initialise at the end of the `__init__` 
        method of the task class inheriting from `BaseTask` class.

        Attributes:
            num_trainable_params: (int) mumber of trainable model parameters.
            num_nontrainable_params: (int) mumber of nontrainable model parameters.
            num_total_params: (int) mumber of total model parameters.
        """
        self.set_num_model_params()
        self.print_num_model_params()
 

    def on_logging_start(self) -> None:
        """Hook for start of logging stage.
        """
        # Set the appropriate label and step for the x-axis of the plots.
        self.metrics_logged[self.step_label] = self.global_step
    

    def on_logging_end(self) -> None:
        """Hook for end of logging stage.
        """
        # Push metrics to the logger
        self.logger.log_metrics(self.metrics_logged)
        # Reset metrics dictionary
        self.metrics_logged.clear()


    def update_losses(self, stage: str, losses: dict) -> None:
        """Update loss accumulator with current step losses.

        Args:
            stage: 
            losses: current step losses.
        """
        for key in losses:
            if key not in self.batch_losses[stage]:
                self.batch_losses[stage][key] = []
            self.batch_losses[stage][key].append(losses[key])
            

    def gather_and_store_losses(self, stage: str) -> None:
        """Aggregate losses across steps and distributed processes.

        Args:
            stage: .
        """
        #print(self.global_step, self.batch_losses[stage])
        for key in self.batch_losses[stage]:
            # Aggregate across steps
            losses = torch.stack(self.batch_losses[stage][key])
            # Aggregate across distributed processes
            losses = self.all_gather(losses)
            # Reduce
            losses = losses.mean().item()
            # Store
            self.metrics_logged[key] = losses
        # Reset accumulator
        self.batch_losses[stage].clear()
        #print(self.global_step, self.metrics_logged)


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
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self._num_nontrainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        self._num_total_params = self._num_trainable_params + self._num_nontrainable_params

    
    def print_num_model_params(self) -> None:
        """Print the number of model parameters.
        """
        info_message("Trainable model parameters: {:,d}".format(self.num_trainable_params))
        info_message("Non-trainable model parameters: {:,d}".format(self.num_nontrainable_params))
        info_message("Total model parameters: {:,d}".format(self.num_total_params))


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