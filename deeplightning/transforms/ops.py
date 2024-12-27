from omegaconf import ListConfig, DictConfig
import torch
from torchvision import transforms as T

from deeplightning.transforms.helpers import (
    pair,
    all_none_or_zero,
    all_none_or_one,
)


def CenterCrop(size) -> T.CenterCrop:
    if all_none_or_zero(size):
        return None
    
    return T.CenterCrop(size=size)


def ColorJitter(brightness, contrast, saturation, hue) -> T.ColorJitter:
    if (
        all_none_or_zero(brightness) and 
        all_none_or_zero(contrast) and 
        all_none_or_zero(saturation) and 
        all_none_or_zero(hue)
    ):
        return None
        
    if brightness is None:
        brightness = (0, 0)

    if contrast is None:
        contrast = (0, 0)

    if saturation is None:
        saturation = (0, 0)

    if hue is None:
        hue = (0, 0)

    return T.ColorJitter(
        brightness = brightness, 
        contrast = contrast, 
        saturation = saturation, 
        hue = hue,
    )


def Normalize(mean, std) -> T.Normalize:
    if all_none_or_zero(mean) and all_none_or_one(std):
        return None
    
    return T.Normalize(mean=mean, std=std)


def Perspective(distortion_scale, p) -> T.RandomPerspective:
    return T.RandomPerspective(
        distortion_scale=distortion_scale, 
        p=p,
    )


def RandomAffine(degrees, translate, scale, shear) -> T.RandomAffine:
    if (
        all_none_or_zero(degrees) and
        all_none_or_zero(translate) and
        all_none_or_zero(scale) and
        all_none_or_zero(shear)
    ):
        return None

    if degrees is not None:
        degrees = tuple(degrees)
    else:
        raise ValueError("Parameter 'degrees' cannot be None.")
    
    if translate is not None:
        translate = tuple(translate)
    
    if scale is not None:
        scale = tuple(scale)
    
    if shear is not None:
        shear = tuple(shear)
    
    return T.RandomAffine(
        degrees = degrees,
        translate = translate,
        scale = scale,
        shear = shear,
)


def RandomCrop(size, padding=None) -> T.RandomCrop:
    if all_none_or_zero(size):
        return None
    return T.RandomCrop(size=size, padding=padding)
        

def RandomHorizontalFlip(p) -> T.RandomHorizontalFlip:
    if all_none_or_zero(p):
        return None

    return T.RandomHorizontalFlip(p)


def RandomResizedCrop(size, scale, ratio) -> T.RandomResizedCrop:
    raise NotImplementedError()
    

def RandomRotation(degrees) -> T.RandomRotation:
    if all_none_or_zero(degrees):
        return None

    return T.RandomRotation(degrees=degrees)


def RandomVerticalFlip(p: float) -> T.RandomVerticalFlip:
    if all_none_or_zero(p):
        return None
    
    return T.RandomVerticalFlip(p)


def Resize(size) -> T.Resize:
    if all_none_or_zero(size):
        return None
    
    return T.Resize(size, antialias=True)


class RoundToInteger(torch.nn.Module):
    """Converts tensor to integer by rounding.
    This is useful after resizing segmentation masks as the interpolation 
    method used in the resizing transform introduces non-integer values.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.round(x).long()
        
    def __repr__(self):
        return "RoundToInteger()"


def RoundToInt():        
    return RoundToInteger()


def ToTensor() -> T.ToTensor:
    return T.ToTensor()


TRANSFORMS_MAP = {
    "centercrop": CenterCrop,
    "colorjitter": ColorJitter,
    "normalize": Normalize,
    #"pad": Pad,
    #"padsquare": PadSquare,
    "perspective": Perspective,
    "randomaffine": RandomAffine,
    "randomcrop": RandomCrop,
    "randomhorizontalflip": RandomHorizontalFlip,
    "randomresizedcrop": RandomResizedCrop,
    "randomrotation": RandomRotation,
    "randomverticalflip": RandomVerticalFlip,
    "resize": Resize,
    "roundtointeger": RoundToInt,
    "totensor": ToTensor,
}
