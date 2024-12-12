from omegaconf import DictConfig
from torchvision import transforms as T
import inspect

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.transforms.ops import __all__ as TransformsDict



def get_num_args(fn):
    args = inspect.getfullargspec(fn).args
    n = len(args)
    if "self" in args:
        n -= 1
    return n


def load_transforms(cfg: DictConfig, subset: str) -> T.Compose:
    """Load data transformations.

    Args:
        cfg: configuration file.
        subset: the data subset for which to load transforms. Transforms are
            read from the configuration field `cfg.data.{subset}_transforms`.
    """

    transforms_field = f"{subset}_transforms"

    trfs = [TransformsDict["totensor"]()]

    if cfg.data[transforms_field] is not None:
        for key in cfg.data[transforms_field].keys():
            fn = TransformsDict[key]
            if get_num_args(fn) == 0:
                transform = fn()
            else:
                params = cfg.data[transforms_field][key]
                transform = fn(params)
            if transform is not None:
                if isinstance(transform, list):
                    for x in transform:
                        trfs.append(x)
                else:
                    trfs.append(transform)
            else:
                warning_message(
                    f"Transform '{key}' present in cfg.data.{transforms_field}' "
                    f"but unused due to unsuitable parameters "
                    f"({cfg.data[transforms_field][key]}).")
        
        print_transforms(transforms_field, trfs)
        return T.Compose(trfs)


def print_transforms(transforms_field, transforms):
    info_message("{}:".format(transforms_field.upper()))
    for i, k in enumerate(transforms):
        info_message(f"  ({i+1}) {k}")
