from torchvision import transforms as T
import omegaconf 

from deeplightning.data.transforms.helpers import is_false_or_none


def HorizontalFlip(p: float):
    """ Horizontal Flip transformation.
    
    Parameters
    ----------
    p : configuration parameter for this transform, the flip 
        probability. This should be the subconfig of the transform, 
        that is `subcfg=cfg.data.transforms.hflip`, where `cfg` is 
        the full experiment config.

    """

    if is_false_or_none(p):
        return None

    assert p >= 0 and p <= 1, "HorizontalFlip transform: flip parameter " \
        "must be between 0 and 1 but found ({})".format(p)

    if p == 0:
        return None

    return T.RandomHorizontalFlip(p)


def VerticalFlip(p: float):
    """ Vertical Flip transformation.
    
    Parameters
    ----------
    p : configuration parameter for this transform, the flip 
        probability. This should be the subconfig of the transform, 
        that is `subcfg=cfg.data.transforms.vflip`, where `cfg` is 
        the full experiment config.

    """
    
    if is_false_or_none(p):
        return None

    assert p >= 0 and p <= 1, "VerticalFlip transform: flip parameter " \
        "must be between 0 and 1 but found ({})".format(p)

    if p == 0:
        return None

    return T.RandomVerticalFlip(p)
