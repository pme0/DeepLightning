from torchvision import transforms as T

from deeplightning.core.dlconfig import DeepLightningConfig
from deeplightning.transforms.ops import TRANSFORMS_MAP
from deeplightning.utils.messages import info_message, warning_message
from deeplightning.utils.python_utils import get_num_args


def load_transforms(
    cfg: DeepLightningConfig,
    subset: str,
) -> T.Compose:
    """Load data transformations.

    Args:
        cfg: configuration file.
        subset: the data subset for which to load transforms. Transforms are
            read from the configuration field `cfg.data.{subset}_transforms`.
    """

    trfs = [TRANSFORMS_MAP["totensor"]()]

    trfs_elems = getattr(cfg.data.transforms, subset)
    if trfs_elems:
        for key in trfs_elems:
            fn = TRANSFORMS_MAP[key]
            if get_num_args(fn) == 0:
                transform = fn()
            else:
                params = trfs_elems[key]
                if isinstance(params, dict):
                    transform = fn(**params)
                else:
                    transform = fn(params)
        
            if transform is not None:
                if isinstance(transform, list):
                    trfs.extend(transform)
                else:
                    trfs.append(transform)
            else:
                warning_message(
                    f"Transform '{key}' present in cfg.data.transforms.{subset}' "
                    f"but unused due to unsuitable parameters "
                    f"({cfg.data.transforms[subset][key]}).")
            
    print_transforms(subset, trfs)
    return T.Compose(trfs)


def print_transforms(subset, transforms):
    info_message("{}:".format(subset.upper()))
    for i, k in enumerate(transforms):
        info_message(f"  ({i+1}) {k}")