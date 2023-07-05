from typing import Union, Tuple
from torchvision import transforms as T
import omegaconf

from deeplightning.data.transforms.helpers import is_false_or_none, is_all_false_or_none


pair = lambda x: x if isinstance(x, tuple) or isinstance(x, omegaconf.listconfig.ListConfig) else (x, x)


def CenterCrop(s: Union[int, Tuple[int, int]]):
    """CenterCrop transformation.

    Parameters
    ----------
    s : configuration parameters for this transform, the crop 
        dimensions. This should be the subconfig of the transform, 
        that is `s=cfg.data.transforms.centercrop`, where `cfg` is the 
        full experiment config

    """     
    s = pair(s)

    if is_false_or_none(s):
        return None
    
    if is_all_false_or_none(s):
        return None

    assert len(s) == 2, f"`s` must have length 2, found length {len(s)}"
    assert isinstance(s, tuple) or isinstance(s, omegaconf.listconfig.ListConfig), "`s` must be tuple or omegaconf.listconfig.ListConfig"
    assert isinstance(s[0], int), "`s[0]` must be int"
    assert isinstance(s[1], int), "`s[1]` must be int"

    return T.CenterCrop(size=s.size)
