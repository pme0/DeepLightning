import argparse
import wandb

from deeplightning.utilities.cleanup import clean_phantom_folder
from deeplightning.utilities.messages import info_message, warning_message, error_message
from deeplightning.config.load import load_config
from deeplightning.init.initializers import init_everything


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/base.yaml", help="")
    args = parser.parse_args()
    return args


def main(cfg):

    model, data, trainer = init_everything(cfg)
    
    trainer.fit(
        model = model,
        datamodule = data,
        ckpt_path = cfg.train.ckpt_resume_path,
    )


if __name__ == "__main__":

    args = parse_command_line_arguments()
    cfg = load_config(config_file = args.cfg)
    
    if cfg.logger.log_to_wandb:
        wandb.init(
            project = cfg.logger.project_name,
            notes = "tweak baseline",
            tags = ["baseline", "paper1"]
        )
    
    try:
        model, data, trainer = init_everything(cfg)

        if cfg.modes.train:
            info_message("Performing training and validation.")
            trainer.fit(
                model = model,
                datamodule = data,
                ckpt_path = cfg.train.ckpt_resume_path,
            )
            if cfg.modes.test:
                info_message(f"Performing testing with last trained model '{trainer.logger.artifact_path}/last.ckpt'.")
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

        info_message("Artifact storage path: {}".format(trainer.logger.artifact_path))

    except KeyboardInterrupt as e:
        warning_message("Interrupted by user.")
        info_message("Artifact storage path: {}".format(trainer.logger.artifact_path))
    finally:
        if cfg.logger.log_to_wandb:
            wandb.finish()
            
