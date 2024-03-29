from typing import Tuple, List, Union
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision import transforms
import lightning as pl

from deeplightning.utils.init.imports import init_obj_from_config
from deeplightning.utils.messages import info_message
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
        #self.device = 

        self.loss = init_obj_from_config(cfg.model.loss)
        self.model = init_obj_from_config(cfg.model.network)
        self.d_optimizer = init_obj_from_config(cfg.model.optimizer.discriminator, self.model.discriminator.parameters())
        self.g_optimizer = init_obj_from_config(cfg.model.optimizer.generator, self.model.generator.parameters())
        self.d_scheduler = init_obj_from_config(cfg.model.scheduler.discriminator, self.d_optimizer)
        self.g_scheduler = init_obj_from_config(cfg.model.scheduler.generator, self.g_optimizer)
       
        #self.maxlen = len(str(self.cfg.train.num_epochs))
        self.train_what == "discriminator"

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info_message("Trainable parameters: {:,d}".format(trainable_params))

    def forward(self, x: Tensor):
        return self.model(x)

    def configure_optimizers(self):
        return (
            {
                "optimizer": self.d_optimizer,
                "lr_scheduler": {
                    "scheduler": self.d_scheduler,
                    "interval": self.cfg.model.scheduler.discriminator.call.interval,
                    "frequency": self.cfg.model.scheduler.discriminator.call.frequency
                }
            },
            {
                "optimizer": self.g_optimizer,
                "lr_scheduler": {
                    "scheduler": self.g_scheduler,
                    "interval": self.cfg.model.scheduler.generator.call.interval,
                    "frequency": self.cfg.model.scheduler.generator.call.frequency
                }
            }
        )

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


    def _find_what_to_train(self, step: int, optim_steps_d: int, optim_steps_g: int) -> str:
        """
        Find which subnetwork to optimize (discriminator or generator) given the 
        current step. This assumes that we start by optimizing the discriminator 
        and then alternate between discriminator and generator according to the 
        schedule implied by the number of optimization steps taken by each.

        Parameters
        ----------
        step : the current training step, it should be `self.global_step`
        optim_steps_d : number of consecutive steps to optimize the discriminator
        optim_steps_g : number of consecutive steps to optimize the generator
        """
        r = step % (optim_steps_d + optim_steps_g)
        if r < optim_steps_d:
            return "discriminator"
        else:
            return "generator"


    def discriminator_step(self, x):
        """Training step for discriminator network.

        1. For real images
            1.1. compute probabilities of real images
            1.2. compute loss on real images
        2. For fake images:
            2.1. generate fake images
            2.2. compute probabilities of fake images
            2.3. compute loss on fake images
        3. Add losses

        """

        # the targets are computed on-the-fly because the last batch may
        # have a different number of images. This is computationally
        # inefficient, alternatively we can drop the last batch in the
        # dataloader (`torch.utils.data.DataLoader(..., drop_last=True)`)
        # and reuse the pre-computed targets.
        self.real_targets = torch.ones(x.shape[0], device=self.device)
        self.fake_targets = torch.zeros(x.shape[0], device=self.device)
        
        # real images
        d_probs = torch.squeeze(self.model.discriminator(x))
        loss_real = self.loss(d_probs, self.real_targets)

        # fake images
        z = torch.randn(x.shape[0], 100, device=self.device)
        generated_imgs = self.model.generator(z)
        d_probs = torch.squeeze(self.model.discriminator(generated_imgs))
        loss_fake = self.loss(d_probs, self.fake_targets)

        return loss_real + loss_fake

    def generator_step(self, x):
        """Training step for generator network.
        """
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        train_this = self._find_what_to_train(
            step = self.global_step, 
            optim_steps_d = self.cfg.model.optimizer.discriminator.steps, 
            optim_steps_g = self.cfg.model.optimizer.generator.steps,
        )

        # train discriminator
        #if optimizer_idx == 0:
        if train_this == "discriminator":
            loss = self.discriminator_step(batch["images"])
    
        # train generator
        #if optimizer_idx == 1:
        if train_this == "generator":
            loss = self.generator_step(batch["images"])

        return {"loss": loss}

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
