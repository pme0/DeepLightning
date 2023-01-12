import os
import sys
import argparse

from deeplightning.utils.messages import info_message, warning_message, error_message
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

    os.system("python deeplightning/inference/object_detection.py --input_path {} --model_cfg {} --model_ckpt {}".format(
        cfg.input_path, cfg.model_cfg, cfg.model_ckpt,
    ))