from typing import Union, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import omegaconf

from deeplightning.transforms.helpers import is_false_or_none, is_all_false_or_none, is_all_constant


def Pad(p):
    """Pad transformation. Pad borders by a number of pixels.

    Args:
    p: configuration parameters for this transform, the 
        paddig values. This should be the subconfig of the 
        padding, that is `subcfg=cfg.data.transforms.pad`, 
        where `cfg` is the full experiment config.
    """   

    if is_false_or_none(p):
        return None
        
    if not isinstance(p, omegaconf.listconfig.ListConfig):
        raise ValueError(f"Transforms: padding parameter ({p}) is not list")
        
    if len(p) != 4:
        raise ValueError("Transforms-Padding: padding parameter list has length {len(cfg)} but should have length 4")
        
    if not np.all(np.array([isinstance(x, int) for x in p])):
        raise ValueError(f"Transforms: padding values ({p}) must be integers")

    if is_all_constant(p, 0):
        return None

    return T.Pad(tuple(p))



def PadSquare(p):
    """Pad Square transformation. Pad borders to a square shape.

    Args:
        p: configuration parameters for this transform, the 
            paddig values. This should be the subconfig of the 
            padding, that is `subcfg=cfg.data.transforms.squarepad`, 
            where `cfg` is the full experiment config.

    """

    if is_false_or_none(p):
        return None

    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp, vp)
            return F.pad(image, padding, 0, 'constant')

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"
    
    return SquarePad()