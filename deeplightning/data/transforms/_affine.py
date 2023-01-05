import omegaconf
from torchvision import transforms as T

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.data.transforms.helpers import is_false_or_none, is_all_false_or_none, is_all_constant


def Affine(subcfg: omegaconf.OmegaConf):
    """Affine transformation.

    Parameters
    ----------
    subcfg : configuration parameters for this transform. This 
        should be the subconfig of the transform, that is 
        `subcfg=cfg.data.transforms.affine`, where `cfg` 
        is the full experiment config. It must contain 
        the following fields:
            - `subcfg.degrees`
            - `subcfg.translate`
            - `subcfg.scale`

    """   

    if is_false_or_none(subcfg):
        return None
        
    if is_all_false_or_none(subcfg):
        return None

    assert isinstance(subcfg.degrees, omegaconf.listconfig.ListConfig), f"'degrees' ({subcfg.degrees}) must be list"
    assert isinstance(subcfg.translate, omegaconf.listconfig.ListConfig), f"'translate' ({subcfg.translate}) must be list"
    assert isinstance(subcfg.scale, omegaconf.listconfig.ListConfig), f"'scale' ({subcfg.scale}) must be list"

    if is_all_constant(subcfg.degrees, 0) and is_all_constant(subcfg.translate, 0) and is_all_constant(subcfg.scale, 0):
        return None

    return T.RandomAffine(
        degrees = tuple(subcfg.degrees),
        translate = tuple(subcfg.translate),
        scale = tuple(subcfg.scale))
