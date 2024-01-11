import omegaconf
from torchvision import transforms as T

from deeplightning.transforms.helpers import is_false_or_none, is_all_false_or_none, is_all_constant


def RandomCrop(subcfg: omegaconf.OmegaConf):
    """ Random Crop transformation.
    
    Args:
        subcfg: configuration parameters for this transform. This should 
            be the subconfig `subcfg=cfg.data.transforms.crop`, where `cfg` 
            is the full experiment config. It must contain the following 
            fields:
                - `subcfg.size`
                - `subcfg.pad` 
    """

    if is_false_or_none(subcfg):
        return None

    if is_all_false_or_none(subcfg):
        return None

    assert isinstance(subcfg.size, int) or isinstance(subcfg.size, omegaconf.listconfig.ListConfig)
    assert isinstance(subcfg.pad, int) or isinstance(subcfg.pad, omegaconf.listconfig.ListConfig)

    return T.RandomCrop(size=subcfg.size, padding=subcfg.pad)
        
    

def RandomResizedCrop():
    """
    """ 

    raise NotImplementedError("ensure no conflict when both `resize` and `resizedcrop` are provided")

    

