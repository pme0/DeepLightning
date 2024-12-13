import omegaconf
from torchvision import transforms as T

from deeplightning.transforms.helpers import is_false_or_none, is_all_false_or_none, is_all_constant


def ColorJitter(subcfg: omegaconf.OmegaConf):
    """ Color Jitter transformation.
    
    Args:
        subcfg: configuration parameters for this transform. This 
            should be the subconfig of the transform, that is 
            `subcfg=cfg.data.transforms.colorjitter`, where `cfg` 
            is the full experiment config. It must contain 
            the following fields:
                - `subcfg.brightness`
                - `subcfg.contrast`
                - `subcfg.saturation`
                - `subcfg.hue`
    """

    if is_false_or_none(subcfg):
        return None
    
    if is_all_false_or_none(subcfg):
        return None

    if is_all_constant(subcfg, 0):
        return None

    assert isinstance(subcfg.brightness, omegaconf.listconfig.ListConfig), f"'brightness' ({subcfg.brightness}) must be list"
    assert isinstance(subcfg.contrast, omegaconf.listconfig.ListConfig), f"'contrast' ({subcfg.contrast}) must be list"
    assert isinstance(subcfg.saturation, omegaconf.listconfig.ListConfig), f"'saturation' ({subcfg.scalesaturation}) must be list"
    assert isinstance(subcfg.hue, omegaconf.listconfig.ListConfig), f"'scale' ({subcfg.hue}) must be list"

    return T.ColorJitter(
        brightness=subcfg.brightness, 
        contrast=subcfg.contrast, 
        saturation=subcfg.saturation, 
        hue=subcfg.hue)