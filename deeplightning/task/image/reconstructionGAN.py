from typing import Tuple, List, Union
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision import transforms
import pytorch_lightning as pl

from deeplightning.init.imports import init_obj_from_config
from deeplightning.utilities.messages import info_message
from deeplightning.trainer.gather import gather_on_step, gather_on_epoch


class ImageReconstructionGAN(pl.LightningModule):
    """ Task module for Image Reconstruction. 

    LOGGING: manual logging `self.logger.log()` is used. This
    is more flexible as PyTorchLightning automatic logging 
    `self.log()`) only allows scalars, not histograms, images, etc.
    Additionally, auto-logging doesn't log at step 0, which is useful.

    """
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.num_tokens = cfg.model.network.params.num_tokens
        self.kl_weight = cfg.model.network.params.kl_div_loss_weight

        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.optimizer = init_obj_from_config(cfg.model.optimizer, self.model.parameters())
        self.scheduler = init_obj_from_config(cfg.model.scheduler, self.optimizer)
       
        self.maxlen = len(str(self.cfg.train.num_epochs))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info_message("Trainable parameters: {:,d}".format(trainable_params))

    def forward(self, x: Tensor):
        return self.model(x)

    def configure_optimizers(self):
        return ({
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": self.cfg.model.scheduler.call.interval,
                "frequency": self.cfg.model.scheduler.call.frequency
            }
        })

    def _gather_on_step(self, step_outputs, var, average):
        if isinstance(step_outputs, list):
            agg = [step_outputs[i][var] for i in range(len(step_outputs))]
        else:
            agg = step_outputs[var]
        return torch.sum(agg).item() / (len(agg) if average is True else 1.0)

    def _gather_on_epoch(self, epoch_outputs, var, average):
        if isinstance(epoch_outputs, list):
            agg = [epoch_outputs[i][var] for i in range(len(epoch_outputs))]
        else:
            agg = epoch_outputs[var]
        return sum(agg) / (len(agg) if average is True else 1.0)


    def _save_artifact_images(self, step_outputs, phase):
        for img_type in [f"{phase}_original", f"{phase}_reconstruction"]:
            img = transforms.ToPILImage()(step_outputs[-1][img_type][0])
            file = "{}_{}.png".format(img_type, str(self.current_epoch).zfill(self.maxlen))
            self.logger.experiment.log_image(
                image = img, 
                artifact_file = file,
                run_id = self.logger.run_id
            )
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_recon, logits = self(x)
        train_loss = self.loss(x, x_recon, logits)
            
        return {"loss": train_loss, 
                "train_original": x,
                "train_reconstruction": x_recon.detach()
                }

    def training_step_end(self, training_step_outputs: dict):
        """ At step_end, aggregate losses across all devices.
        """
        agg_loss = self._gather_on_step(training_step_outputs, var="loss", average=False)
        self.logger.log_metrics({
            "train_loss": agg_loss,
            "step": self.global_step,
            }, step=self.global_step)
    
    def training_epoch_end(self, training_step_outputs):
        """ At epoch_end, log accuracy on the last batch of data only.
            NOTE: For *Training*, the input to `training_epoch_end` is the output from 
            `training_step`. This is unlike *Validation* where the input to 
            `validation_epoch_end`  is the output from `validation_step_end` and the input 
            to `validation_step_end` is the output from `validation_step` more (intuitive).
            See https://github.com/PyTorchLightning/pytorch-lightning/issues/9811
        """
        self._save_artifact_images(training_step_outputs, phase="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_recon, logits = self(x)
        val_loss = self.loss(x, x_recon, logits)

        return {"val_loss": val_loss, 
                "val_original": x,
                "val_reconstruction": x_recon.detach()
                }

    def validation_step_end(self, validation_step_outputs):
        """ At step_end, aggregate losses across all devices.
        """
        agg_loss = self._gather_on_step(validation_step_outputs, var="val_loss", average=False)
        return {"val_loss": agg_loss,
                "val_original": validation_step_outputs["val_original"][0],
                "val_reconstruction": validation_step_outputs["val_reconstruction"][0]
                }     

    def validation_epoch_end(self, validation_epoch_outputs):
        """ At epoch_end, aggregate losses across all steps and log the average.
        """
        agg_loss = self._gather_on_epoch(validation_epoch_outputs, var="val_loss", average=True)
        self.logger.log_metrics({"val_loss": agg_loss}, step=self.global_step)
        self._save_artifact_images(validation_epoch_outputs, phase="val")
