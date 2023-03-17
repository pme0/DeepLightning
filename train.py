import sys
import argparse
import wandb
from omegaconf import OmegaConf

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

    # Initialise model, dataset, trainer
    model, data, trainer = init_everything(cfg)
    
    # Update config - it is updated in `init_logger` inside `DLTrainer`
    cfg = trainer.cfg
    config_print(OmegaConf.to_yaml(cfg))

    try:

        if cfg.modes.train:
            info_message("Performing training and validation.")
            trainer.fit(
                model = model,
                datamodule = data,
                ckpt_path = cfg.train.ckpt_resume_path,
            )
            if cfg.modes.test:
                info_message(f"Performing testing with last trained model '{trainer.logger_.artifact_path}/last.ckpt'.")
                trainer.test(
                    model = model,
                    #ckpt_path = "best",
                    datamodule = data,
                )
        else:
            if cfg.modes.test:
                info_message(f"Performing testing with pretrained model '{cfg.test.ckpt_test_path}'.")
                #https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#testing
                trainer.test(
                    model = model,
                    ckpt_path = cfg.test.ckpt_test_path,
                    datamodule = data,
                )

    except KeyboardInterrupt as e:

        warning_message("Interrupted by user.")
        sys.exit()

    finally:
        
        if cfg.logger.log_to_wandb:
            # wandb logger
            info_message("Artifact storage path: {}".format(trainer.logger_.artifact_path))
            wandb.finish()
        else:
            # mlflow logger
            info_message("Artifact storage path: {}".format(trainer.logger.artifact_path))
            
