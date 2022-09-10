from deeplightning.utilities.messages import info_message, warning_message
from torchvision import transforms as T


def get_transforms(DataTransforms, cfg, subset):
    trfs = []
    if cfg.data[subset] is None:
        return trfs.append(DataTransforms["totensor"])
    for k in cfg.data[subset].keys():
        p = cfg.data[subset][k]
        t = DataTransforms[k](p)
        if t is not None:
            if isinstance(t, list):
                for x in t:
                    trfs.append(x)
            else:
                trfs.append(t)
        else:
            warning_message(f"Transform '{k}' present in .cfg.data.{subset}' but unused due to unsuitable parameters ({cfg.data[subset][k]})")
    
    info_message("{}:".format(subset.upper()))
    for i, k in enumerate(trfs):
        info_message(f"  ({i+1}) {k}")

    return T.Compose(trfs)
