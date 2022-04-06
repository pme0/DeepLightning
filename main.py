import argparse

from deeplightning.utilities.cleanup import clean_phantom_folder
from deeplightning.config.load import load_config
from deeplightning.init.initializers import init_everything


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="")
    args = parser.parse_args()
    return args


def main(config):

    model, data, trainer = init_everything(config)
    
    trainer.fit(
        model = model,
        datamodule = data,
        ckpt_path = config.train.ckpt_resume_path,
    )


if __name__ == "__main__":

    args = parse_command_line_arguments()
    config = load_config(config_file = args.config)
    
    try:
        main(config)
    except KeyboardInterrupt as e:
        print("Interrupted by user.")
    finally:
        clean_phantom_folder()
