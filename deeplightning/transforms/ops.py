from omegaconf import ListConfig, DictConfig
from torchvision import transforms as T

from deeplightning.transforms.helpers import (
    pair,
    none_or_zero, 
    all_none_or_zero,
)


def CenterCrop(
    size: ListConfig | None,
):
     
    if all_none_or_zero(size):
        return None

    return T.CenterCrop(
        size = size,
    )


def ColorJitter(
    brightness: None | ListConfig = None,
    contrast: None | ListConfig = None,
    saturation: None | ListConfig = None,
    hue: None | ListConfig = None,
):

    if (
        all_none_or_zero(brightness)
        and all_none_or_zero(contrast)
        and all_none_or_zero(saturation)
        and all_none_or_zero(hue)
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



def RandomAffine(
    degrees: ListConfig | None, 
    translate: ListConfig | None = None, 
    scale: ListConfig | None = None, 
    shear: ListConfig | None = None, 
) -> T.RandomAffine:
    
    if (
        all_none_or_zero(degrees)
        and all_none_or_zero(translate)
        and all_none_or_zero(scale)
        and all_none_or_zero(shear)
    ):
        return None

    if degrees is not None:
        degrees = tuple(degrees)
    else:
        raise ValueError("Parameter `degrees` cannot be None.")
    
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


def Resize(size):

    if all_none_or_zero(size):
        return None
    
    return T.Resize(size, antialias=True)


def ToTensor():
    return T.ToTensor()



#======================

'''
from deeplightning.transforms._crop import RandomCrop, RandomResizedCrop
from deeplightning.transforms._flip import HorizontalFlip, VerticalFlip
from deeplightning.transforms._pad import Pad, PadSquare
from deeplightning.transforms._perspective import Perspective
from deeplightning.transforms._resize import Resize
from deeplightning.transforms._rotation import Rotation
'''

from deeplightning.transforms._normalize import Normalize
from deeplightning.transforms._totensor import ToTensor
from deeplightning.transforms._round import RoundToInteger


__all__ = {
    "centercrop": CenterCrop,
    "colorjitter": ColorJitter,
    #"horizontalflip": HorizontalFlip,
    "normalize": Normalize,
    #"pad": Pad,
    #"padsquare": PadSquare,
    #"perspective": Perspective,
    "randomaffine": RandomAffine,
    #"randomcrop": RandomCrop,
    #"randomresizedcrop": RandomResizedCrop,
    "resize": Resize,
    #"rotation": Rotation,
    "roundtointeger": RoundToInteger,
    "totensor": ToTensor,
    #"verticalflip": VerticalFlip,
}