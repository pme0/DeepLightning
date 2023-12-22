from omegaconf import OmegaConf
from torchvision import transforms as T

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
}


def load_transforms(
    cfg: OmegaConf, 
    subset: str, 
    #transforms_dict: dict,
    ) -> T.Compose:
    """Load data transformations.

    Parameters
    ----------
    cfg : the full experiment config

    subset : the data subset for which to load the transforms. 
        It must be either "train" or "test"
    
    """

    if subset == "train":
        transforms_field = "train_transforms"
    elif subset == "test":
        transforms_field = "test_transforms"
    else:
        raise ValueError("`subset` must be either 'train' or 'test'.")

    trfs = [__TransformsDict__["totensor"]()]

    if transforms_field not in cfg.data:
        print_transforms(transforms_field, trfs)
        return T.Compose(trfs)

    if cfg.data[transforms_field] is not None:
        for key in cfg.data[transforms_field].keys():
            params = cfg.data[transforms_field][key]
            transform = __TransformsDict__[key](params)
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
