from typing import Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
import lightning as pl

from deeplightning.init.imports import init_obj_from_config
from deeplightning.init.initializers import init_metrics
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch
from deeplightning.utils.messages import info_message



class BaseTask(pl.LightningModule):
    """Base task module.

    Notes:
        logging: manual logging `self.logger.log()` is used. This is more 
            flexible as Lightning automatic logging `self.log()`) only 
            allows scalars, not histograms, images, etc./ Additionally, 
            auto-logging doesn't log at step 0, which is useful.
        hooks: For *training*, the input to `training_epoch_end()` is the 
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
        sanity_check: (bool) Lightning performs a partial validation epoch
            at the start, to ensure no issues at the validation stage. The 
            attribute `sanity_check` is set to `True` initially and set to 
            `False` after the sanity check run is complete. This is to
            prevent logging during that preliminary run.
    """
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg  #TODO check if this contains logger runtime params
        self.step_label = "iteration"
        self.sanity_check = True
    
    def on_task_init_end(self) -> None:
        """Additional attributes to initialise at the end of the `__init__` 
        method of the class that inherits from this `BaseTask` class.

        Attributes:
            num_trainable_params: (int) mumber of trainable model parameters.
            num_nontrainable_params: (int) mumber of nontrainable model parameters.
            num_total_params: (int) mumber of total model parameters.
        """
        self.set_num_model_params()
        self.print_num_model_params()
 

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