from torchvision import transforms as T
import omegaconf
from deeplightning.transforms.helpers import is_false_or_none, is_all_false_or_none, is_all_constant


def Rotation(degrees: omegaconf.OmegaConf):
    """Rotation transformation.

    Args:
        degrees: configuration parameters for this transform. This 
            should be the subconfig of the transform, that is 
            `subcfg=cfg.data.transforms.rotation`, where `cfg` 
            is the full experiment config.
    """
     
    if is_false_or_none(degrees):
        return None
    
    assert isinstance(degrees, omegaconf.listconfig.ListConfig), f"'degrees' ({degrees}) must be list, found {type(degrees)}"
    
    if is_all_constant(degrees, 0):
        return None

    return T.RandomRotation(degrees = degrees)

