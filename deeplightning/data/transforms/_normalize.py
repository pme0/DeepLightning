import omegaconf
from torchvision import transforms as T

from deeplightning.data.transforms.helpers import is_false_or_none, is_all_false_or_none, is_all_constant


def Normalize(subcfg: omegaconf.OmegaConf):
    """Normalize transformation.

    use `deeplightning.utils.data.compute_dataset_mean_and_stdev()` to
    obtain the dataset statistics

    Parameters
    ----------
    subcfg : configuration parameters for this transform. This should
        be the subconfig `subcfg=cfg.data.transforms.normalize`, 
        where `cfg` is the full experiment config. It must contain 
        the following fields:
            - `subcfg.mean`
            - `subcfg.std`

    """

    if is_false_or_none(subcfg):
        return None

    if is_all_false_or_none(subcfg):
        return None
     
    assert isinstance(subcfg.mean, omegaconf.listconfig.ListConfig), f"'mean' ({subcfg.mean}) must be list"
    assert isinstance(subcfg.std, omegaconf.listconfig.ListConfig), f"'std' ({subcfg.std}) must be list"
        
    if is_all_constant(subcfg.mean, 0) and is_all_constant(subcfg.std, 1):
        return None
        
    # NOTE T.Normalize() expects tensor
    return T.Normalize(mean=subcfg.mean, std=subcfg.std)

