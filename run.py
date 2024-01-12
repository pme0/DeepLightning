import sys
import argparse
import wandb
#import hydra

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.utils.config.load import load_config
from deeplightning.utils.init.initializers import init_lightning_modules


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to YAML configuration file.")
    #parser.add_argument("--config-path", type=str, default="configs", help="Directory of YAML configuration file. Overwrites `config_path` in hydra.main().")
    #parser.add_argument("--config-name", type=str, help="Filename of YAML configuration file. Overwrites `config_name` in hydra.main().")
    args = parser.parse_args()
    return args


def train_hook(cfg, trainer, model, data):
    ckpt_path = cfg.train.ckpt_resume_path
    if ckpt_path is None:
        info_message("Starting training from scratch.")
    else:
        info_message(f"Resuming training from checkpoint '{ckpt_path}'.")
    trainer.fit(
        model = model,
        datamodule = data,
        ckpt_path = ckpt_path)
    

def test_ckpt_hook(cfg, trainer, model, data):
    ckpt_path = cfg.test.ckpt_test_path
    info_message(f"Starting testing from checkpoint '{ckpt_path}'.")
    trainer.test(
        model = model,
        ckpt_path = ckpt_path,
        datamodule = data)
    

def test_best_hook(cfg, trainer, model, data):
    info_message("Starting testing with the best model.")
    trainer.test(
        model = model,
        ckpt_path = "best",
        datamodule = data)


#@hydra.main(version_base=None, config_path="", config_name="")
def main():

    args = parse_command_line_arguments()

    # Load config
    # NOTE The following config is incomplete. The complete one --- which 
    # includes logger runtime parameters like artifact path --- is updated 
    # inside the LightningModule (from where is gets printed and logged) and 
    # retrieved bellow.
    cfg = load_config(config_file = args.cfg)

    # Initialise: model, dataset, trainer
    model, data, trainer = init_lightning_modules(cfg)

    # Retrieve config augmented with runtime parameters
    cfg = trainer.cfg

    # run training & testing
    try:
        if cfg.modes.train:
            train_hook(cfg, trainer, model, data)
            if cfg.modes.test:
                test_best_hook(cfg, trainer, model, data)
        else:
            if cfg.modes.test:
                test_ckpt_hook(cfg, trainer, model, data)

    except KeyboardInterrupt as e:
        warning_message("Interrupted by user.")
        sys.exit()

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()