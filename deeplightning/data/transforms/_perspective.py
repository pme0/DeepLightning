from omegaconf import OmegaConf
from torchvision import transforms as T

from deeplightning.data.transforms.helpers import is_false_or_none, is_all_false_or_none


def Perspective(subcfg: OmegaConf):
    """Perspective transformation.

    Parameters
    ----------
    subcfg : configuration parameters for this transform. This 
        should be the subconfig of the transform, that is 
        `subcfg=cfg.data.transforms.perspective`, where `cfg` 
        is the full experiment config. It must contain 
        the following fields:
            - `subcfg.distortion_scale`
            - `subcfg.p`

    """

    if is_false_or_none(subcfg):
        return None
    
    if is_all_false_or_none(subcfg) or subcfg.p == 0:
        return None
    
    return T.RandomPerspective(distortion_scale=subcfg.distortion_scale, p=subcfg.p)

