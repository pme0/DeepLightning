from deeplightning.utilities.messages import info_message, warning_message
import numpy as np
from torchvision import transforms as T


def print_transforms(subset, trfs):
    info_message("{}:".format(subset.upper()))
    for i, k in enumerate(trfs):
        info_message(f"  ({i+1}) {k}")

def get_transforms(DataTransforms, cfg, field):
    trfs = []
    if cfg.data[field] is None:
        return trfs.append(DataTransforms["totensor"])
    for k in cfg.data[field].keys():
        p = cfg.data[field][k]
        t = DataTransforms[k](p)
        if t is not None:
            if isinstance(t, list):
                for x in t:
                    trfs.append(x)
            else:
                trfs.append(t)
        else:
            warning_message(f"Transform '{k}' present in .cfg.data.{field}' but unused due to unsuitable parameters ({cfg.data[field][k]})")
    print_transforms(field, trfs)
    return T.Compose(trfs)
