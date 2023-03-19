from typing import Tuple
from omegaconf import OmegaConf
from torch import Tensor
import pytorch_lightning as pl

from deeplightning.config.load import log_config
from deeplightning.init.imports import init_obj_from_config
from deeplightning.init.initializers import init_metrics, init_logger
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch
from deeplightning.utils.messages import info_message, config_print
from deeplightning.utils.registry import __MetricsRegistry__, __HooksRegistry__



class TaskModule(pl.LightningModule):
    """ Task module for Image Classification. 

    LOGGING: manual logging `self.logger.log()` is used. This
    is more flexible as PyTorchLightning automatic logging 
    `self.log()`) only allows scalars, not histograms, images, etc.
    Additionally, auto-logging doesn't log at step 0, which is useful.

    Parameters
    ----------
    cfg : yaml configuration object
    
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.num_classes = cfg.model.network.params.num_classes
        self.classif_task = "binary" if self.num_classes == 2 else "multiclass"

        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)
        
        # by default `pl.Trainer()` has an attribute `logger` so name 
        # this custom one `logger_` to avoid conflicts
        self.logger_ = init_logger(cfg)
        print('logger_', self.logger_)
        print('vars(logger_)', vars(self.logger_))
        
        # update config with logger runtime parameters, and print
        self.cfg = self.logger_.cfg
        config_print(OmegaConf.to_yaml(cfg))
        log_config(cfg=cfg, path=self.logger_.artifact_path)


        # PyTorch-Lightning performs a partial validation epoch to ensure that
        # everything is correct. Use this to avoid logging metrics to W&B for that 
        self.sanity_check = True

        # Initialise metrics to track during training
        self.metrics = init_metrics(cfg)

        # Initialise label to track metrics against
        self.step_label = "iteration"

        # Define hook functions
        # to make the hooks bound to the class (so that they can access class attributes 
        #  using `self.something`), the assignment must specify the class name as follows:
        # `ClassName.fn = my_fn` rather than `self.fn = my_fn`
        TaskModule._training_step = __HooksRegistry__[cfg.task]["training_step"]
        TaskModule._training_step_end = __HooksRegistry__[cfg.task]["training_step_end"]
        TaskModule._training_epoch_end = __HooksRegistry__[cfg.task]["training_epoch_end"]
        TaskModule._validation_step = __HooksRegistry__[cfg.task]["validation_step"]
        TaskModule._validation_step_end = __HooksRegistry__[cfg.task]["validation_step_end"]
        TaskModule._validation_epoch_end = __HooksRegistry__[cfg.task]["validation_epoch_end"]
        TaskModule._test_step = __HooksRegistry__[cfg.task]["test_step"]
        TaskModule._test_step_end = __HooksRegistry__[cfg.task]["test_step_end"]
        TaskModule._test_epoch_end = __HooksRegistry__[cfg.task]["test_epoch_end"]

        # Aggregation utilities
        self.gather_on_step = gather_on_step
        self.gather_on_epoch = gather_on_epoch

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
        """ Hook for `training_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        batch_idx : index of batch

        """
        return self._training_step(batch, batch_idx)


    def training_step_end(self, training_step_outputs):
        """ Hook for `training_step_end`.

        Parameters
        ----------
        training_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `training_step()`.

        """
        self._training_step_end(training_step_outputs)


    def training_epoch_end(self, training_step_outputs):
        """ Hook for `training_epoch_end`.
        
        Parameters
        ----------
        training_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `training_step()`.

        """
        self._training_epoch_end(training_step_outputs)
    

    def validation_step(self, batch, batch_idx):
        """ Hook for `validation_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        batch_idx : index of batch

        """
        return self._validation_step(batch, batch_idx)


    def validation_step_end(self, validation_step_outputs):
        """ Hook for `validation_step_end`.

        Parameters
        ----------
        validation_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `validation_step()`.

        """
        return self._validation_step_end(validation_step_outputs)


    def validation_epoch_end(self, validation_epoch_outputs):
        """ Hook for `validation_epoch_end`.

        Parameters
        ----------
        validation_epoch_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). 
            The output from `validation_step_end()`.

        """
        self._validation_epoch_end(validation_epoch_outputs)


    def test_step(self, batch, batch_idx):
        """ Hook for `test_step`.

        Parameters
        ----------
        batch : object containing the data output by the dataloader. For custom 
            datasets this is a dictionary with keys ["paths", "images", "labels"].
            For torchvision datasets, the function `dictionarify_batch()` is used
            to convert the native format to dictionary format
        
        batch_idx: index of batch.
        
        """
        return self._test_step(batch, batch_idx)


    def test_step_end(self, test_step_outputs):
        """ Hook for `test_step_end`.

        Parameters
        ----------
        test_step_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `test_step()`.

        """
        return self._test_step_end(test_step_outputs)


    def test_epoch_end(self, test_epoch_outputs):
        """ Hook for `test_epoch_end`.

        Parameters
        ----------
        test_epoch_outputs : (dict, list[dict]) metrics 
            dictionary in single-device training, or list of 
            metrics dictionaries in multi-device training (one 
            element per device). The output from `test_step_end()`.
            
        """
        self._test_epoch_end(test_epoch_outputs)