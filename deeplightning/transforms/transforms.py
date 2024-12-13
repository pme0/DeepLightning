from omegaconf import OmegaConf
from torchvision import transforms as T
import inspect

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.transforms._affine import Affine
from deeplightning.transforms._centercrop import CenterCrop
from deeplightning.transforms._colorjitter import ColorJitter
from deeplightning.transforms._crop import RandomCrop, RandomResizedCrop
from deeplightning.transforms._flip import HorizontalFlip, VerticalFlip
from deeplightning.transforms._normalize import Normalize
from deeplightning.transforms._pad import Pad, PadSquare
from deeplightning.transforms._perspective import Perspective
from deeplightning.transforms._resize import Resize
from deeplightning.transforms._rotation import Rotation
from deeplightning.transforms._totensor import ToTensor
from deeplightning.transforms._round import RoundToInteger


__TransformsDict__ = {
    "affine": Affine,
    "centercrop": CenterCrop,
    "colorjitter": ColorJitter,
    "crop": RandomCrop,
    "hflip": HorizontalFlip,
    "normalize": Normalize,
    "pad": Pad,
    "padsquare": PadSquare,
    "perspective": Perspective,
    "resize": Resize,
    "resizedcrop": RandomResizedCrop,
    "rotation": Rotation,
    "roundtointeger": RoundToInteger,
    "totensor": ToTensor,
    "vflip": VerticalFlip,
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
        subset: the data subset for which to load transforms. Transforms are
            read from the configuration field `cfg.data.{subset}_transforms`.
    """

    transforms_field = f"{subset}_transforms"

    trfs = [__TransformsDict__["totensor"]()]

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
                warning_message(
                    f"Transform '{key}' present in cfg.data.{transforms_field}' "
                    f"but unused due to unsuitable parameters "
                    f"({cfg.data[transforms_field][key]}).")
        
        print_transforms(transforms_field, trfs)
        return T.Compose(trfs)


def print_transforms(transforms_field, transforms):
    info_message("{}:".format(transforms_field.upper()))
    for i, k in enumerate(transforms):
        info_message(f"  ({i+1}) {k}")
