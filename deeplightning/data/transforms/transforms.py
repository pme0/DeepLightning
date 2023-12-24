from omegaconf import OmegaConf
from torchvision import transforms as T
import inspect

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.data.transforms._affine import Affine
from deeplightning.data.transforms._crop import RandomCrop, RandomResizedCrop
from deeplightning.data.transforms._centercrop import CenterCrop
from deeplightning.data.transforms._colorjitter import ColorJitter
from deeplightning.data.transforms._flip import HorizontalFlip, VerticalFlip
from deeplightning.data.transforms._normalize import Normalize
from deeplightning.data.transforms._pad import Pad, PadSquare
from deeplightning.data.transforms._perspective import Perspective
from deeplightning.data.transforms._resize import Resize
from deeplightning.data.transforms._rotation import Rotation
from deeplightning.data.transforms._totensor import ToTensor
from deeplightning.data.transforms._round import RoundToInteger


__TransformsDict__ = {
    "affine": Affine,
    "colorjitter": ColorJitter,
    "crop": RandomCrop,
    "hflip": HorizontalFlip,
    "normalize": Normalize,
    "pad": Pad,
    "padsquare": PadSquare,
    "perspective": Perspective,
    "resize": Resize,
    "centercrop": CenterCrop,
    "resizedcrop": RandomResizedCrop,
    "rotation": Rotation,
    "totensor": ToTensor,
    "vflip": VerticalFlip,
    "roundtointeger": RoundToInteger,
}


def get_num_params(fn):
    args = inspect.getfullargspec(fn).args
    n = len(args)
    if "self" in args:
        n -= 1
    return n


def load_transforms(cfg: OmegaConf, subset: str) -> T.Compose:
    """Load data transformations.

    Args:
        cfg: configuration file.
        subset: the data subset for which to load the transforms. It must be 
        either "train", "test", or "mask".
    """

    if subset == "train":
        transforms_field = "train_transforms"
    elif subset == "test":
        transforms_field = "test_transforms"
    elif subset == "mask":
        transforms_field = "mask_transforms"
    else:
        raise ValueError("Invalid subset for transforms")

    trfs = [__TransformsDict__["totensor"]()]

    if transforms_field not in cfg.data:
        print_transforms(transforms_field, trfs)
        return T.Compose(trfs)

    if cfg.data[transforms_field] is not None:
        for key in cfg.data[transforms_field].keys():
            fn = __TransformsDict__[key]
            if get_num_params(fn) == 0:
                transform = fn()
            else:
                params = cfg.data[transforms_field][key]
                transform = fn(params)
            if transform is not None:
                if isinstance(transform, list):
                    for x in transform:
                        trfs.append(x)
                else:
                    trfs.append(transform)
            else:
                warning_message(f"Transform '{key}' present in cfg.data.{transforms_field}' but unused due to unsuitable parameters ({cfg.data[transforms_field][key]})")
        
        print_transforms(transforms_field, trfs)
        return T.Compose(trfs)


def print_transforms(transforms_field, transforms):
    info_message("{}:".format(transforms_field.upper()))
    for i, k in enumerate(transforms):
        info_message(f"  ({i+1}) {k}")
