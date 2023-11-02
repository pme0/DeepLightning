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
    """Base task module

    LOGGING: manual logging `self.logger.log()` is used. This is more 
    flexible as Lightning automatic logging `self.log()`) only allows 
    scalars, not histograms, images, etc./ Additionally, auto-logging 
    doesn't log at step 0, which is useful.

    HOOKS: For *training*, the input to `training_epoch_end()` is the 
    set of outputs from `training_step()`. For *validation*, the input 
    to `validation_epoch_end()` is the output from `validation_step_end()` 
    and the input to `validation_step_end()` is the output from 
    `validation_step()`.
    See https://github.com/PyTorchLightning/pytorch-lightning/issues/9811

    Args
        cfg: yaml configuration object
    
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg  #TODO check if this contains logger runtime params

        # Lightning performs a partial validation epoch to ensure that 
        # everything is correct. Use this to avoid logging during that
        self.sanity_check = True

        # Initialise metrics to track during training
        self.device = torch.device("cuda") if cfg.engine.accelerator == "gpu" else torch.device('cpu')
        self.metrics = init_metrics(cfg, device=self.device)

        # Initialise label to track metrics against
        self.step_label = "iteration"

        # Aggregation utilities
        self.gather_on_step = gather_on_step
        self.gather_on_epoch = gather_on_epoch


    def num_parameters(self):
        """Prints the number of model parameters

        Lightning's model summary does not give the correct number 
        of trainable parameters. See 
        https://github.com/PyTorchLightning/pytorch-lightning/issues/12130
        """
    
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        nontrainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total_params =  trainable_params + nontrainable_params
        
        info_message("Trainable model parameters: {:,d}".format(trainable_params))
        info_message("Non-trainable model parameters: {:,d}".format(nontrainable_params))
        info_message("Total model parameters: {:,d}".format(total_params))
    
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


    def configure_optimizers(self) -> Tuple[dict]:
        raise NotImplementedError
    

    def training_step(self, batch, batch_idx):
        raise NotImplementedError


    def training_step_end(self):
        raise NotImplementedError


    def on_training_epoch_end(self):
        raise NotImplementedError
    

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError


    def validation_step_end(self):
        raise NotImplementedError


    def on_validation_epoch_end(self):
        raise NotImplementedError


    def test_step(self, batch, batch_idx):
        raise NotImplementedError


    def test_step_end(self):
        raise NotImplementedError


    def on_test_epoch_end(self):
        raise NotImplementedError