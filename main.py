import argparse

from deeplightning.utilities.cleanup import clean_phantom_folder
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
    
    try:
        main(cfg)
    except KeyboardInterrupt as e:
        print("Interrupted by user.")
    finally:
        pass
    #clean_phantom_folder()
