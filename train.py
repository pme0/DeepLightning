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
    cfg = load_config(config_file = args.cfg)
    
    # Initialise logger
    # TODO put the following inside init_everything()
    if cfg.logger.log_to_wandb:
        wandb.init(
            project = cfg.logger.project_name,
            notes = cfg.logger.notes,
            tags = cfg.logger.tags,
        )
        logger_run_id = wandb.run.id
        logger_run_name = wandb.run.name
        logger_run_dir = wandb.run.dir
    # add logger params to config
    cfg.logger.runtime = {}  # TODO find better way to create nested keys without create each level in sequence
    cfg.logger.runtime.run_id = logger_run_id
    cfg.logger.runtime.run_name = logger_run_name
    cfg.logger.runtime.run_dir = logger_run_dir
    #cfg = add_logger_params(config_file = args.cfg) TODO encapsulate the above in a function

    # Initialise model, dataset, trainer
    model, data, trainer = init_everything(cfg)
    cfg.logger.runtime.artifact_path = trainer.logger_.artifact_path
    config_print(OmegaConf.to_yaml(cfg))

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
            
