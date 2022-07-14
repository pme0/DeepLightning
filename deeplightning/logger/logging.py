from typing import Any, Union
from omegaconf import OmegaConf
from matplotlib.figure import Figure
import os
import torch
from torch import Tensor
from torchvision import transforms

from deeplightning.utilities.messages import warning_message, error_message


# helpers

__ImageFormats__ = ["bmp", "jpeg", "jpg", "png", "tiff"]


def extension(string: str):
    return string.rsplit(".")[-1]


def check_image_format(name: str):
    if not extension(name) in __ImageFormats__:
        warning_message(
            "Attempting to save an image with extension "
            "('{}'), which may not be a valid image "
            "format ({}).".format(extension(name), __ImageFormats__))


# mains

def log_config(cfg: OmegaConf, path: str) -> None:
    """ Save configuration (.yaml)
    """
    if not OmegaConf.is_config(cfg):
        error_message(
            "Attempting to save a config artifact but the object "
            "provided is not of type omegaconf.dictconfig.DictConfig.")
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    OmegaConf.save(cfg, f = os.path.join(path, "cfg.yaml"))


def log_image(tensor: Tensor, folder: str, name: str):
    """ Save image (.png/.jpeg/...)
    """
    check_image_format(name)
    image = transforms.ToPILImage()(tensor)
    image.save(os.path.join(folder, name))


def log_figure(figure: Figure, folder: str, name: str):
    """ Save matplotlib figure (.png/.jpeg/...)
    """
    check_image_format(name)
    figure.savefig(os.path.join(folder, name))


def log_histogram(tensor: torch.Tensor, folder: str, name: str):
    """ Save matplotlib figure (.png/.jpeg/...)
    """
    check_image_format(name)
    raise NotImplementedError
    # create hist plot from tensor
    #figure = f(tensor)
    #figure.savefig(os.path.join(folder, name))


def log_artifact(self, artifact: Any, artifact_type: str, artifact_path: str) -> None:
    """ Save artifact files.
    """

    if artifact_type == "config":

        # save OmegaConf config to .yaml file
        if not OmegaConf.is_config(artifact):
            error_message(
                "Attempting to save a config artifact but the object "
                "provided is not of type omegaconf.dictconfig.DictConfig.")

        log_config(artifact, artifact_path)


    elif artifact_type == "image":
        
        # check that is PIL image
        raise NotImplementedError


    elif artifact_type == "figure":
        # check that is matplotlib figure
        raise NotImplementedError

    else:
        raise NotImplementedError
