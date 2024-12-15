from omegaconf import DictConfig
from torchvision import transforms as T

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.transforms.ops import __all__ as TRANSFORMS_MAP
from deeplightning.transforms.helpers import get_num_args


def load_transforms(cfg: DictConfig, subset: str) -> T.Compose:
    """Load data transformations.

    Args:
        cfg: configuration file.
        subset: the data subset for which to load transforms. Transforms are
            read from the configuration field `cfg.data.{subset}_transforms`.
    """

    trfs = [TRANSFORMS_MAP["totensor"]()]

    if cfg.data.transforms[subset] is not None:
        for key in cfg.data.transforms[subset]:
            fn = TRANSFORMS_MAP[key]
            if get_num_args(fn) == 0:
                transform = fn()
            else:
                params = cfg.data.transforms[subset][key]
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