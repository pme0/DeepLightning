import sys
import argparse
import wandb
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint

from deeplightning.utils.cleanup import clean_phantom_folder
from deeplightning.utils.messages import info_message, warning_message, error_message, config_print
from deeplightning.config.load import load_config
from deeplightning.init.initializers import init_everything


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/base.yaml", help="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_command_line_arguments()

    # Load config
    cfg = load_config(config_file = args.cfg)
    # [!] this config is incomplete; the complete one, which includes
    #     logger runtime parameters like artifact path is inside the
    #     LightningModule and it gets printed & logged from there
    #TODO find a better way to do this

    # Initialise: model, dataset, trainer
    model, data, trainer = init_everything(cfg)

    # Load config augmented with logger runtime parameters
    cfg = trainer.cfg

    try:

        if cfg.modes.train:
            
            info_message("Performing training and validation.")
            trainer.fit(
                model = model,
                datamodule = data,
                ckpt_path = cfg.train.ckpt_resume_path,)
            
            if cfg.modes.test:

                which_ckpt = None
                if "checkpoint" in trainer.callbacks_dict:
                    if isinstance(trainer.callbacks_dict["checkpoint"], ModelCheckpoint):
                        which_ckpt = "best"

                info_message("Performing testing with {} model".format("best" if which_ckpt == "best" else "last"))
                trainer.test(
                    model = model,
                    ckpt_path = which_ckpt,
                    datamodule = data,)
        else:
            if cfg.modes.test:

                info_message(f"Performing testing with pretrained model '{cfg.test.ckpt_test_path}'.")
                #https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#testing
                trainer.test(
                    model = model,
                    ckpt_path = cfg.test.ckpt_test_path,
                    datamodule = data,)

    except KeyboardInterrupt as e:

        warning_message("Interrupted by user.")
        sys.exit()

    finally:
        
        wandb.finish()

