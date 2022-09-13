from torchvision import transforms as T
import torchvision.transforms.functional as F
import numpy as np

from deeplightning.utilities.messages import info_message, warning_message


def get_transforms(cfg, subset):
    """
    """
    if cfg.data.transforms is None or cfg.data.transforms is False:
        return None

    DataTransforms = {
    "affine": Affine,
    "flip": HorizontalFlip,
    "jitter": ColorJitter,
    "normalize": Normalize,
    "pad": Pad,
    "padsquare": PadSquare,
    "perspective": Perspective,
    "resize": Resize,
    "rotation": Rotation,
    "totensor": ToTensor,
    }

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


def Affine(cfg):
    return T.RandomAffine(
        degrees = tuple(cfg.degrees),
        translate = tuple(cfg.translate),
        scale = tuple(cfg.scale),)


def ColorJitter(cfg):
    return T.ColorJitter(
        brightness=cfg.brightness, 
        contrast=cfg.contrast, 
        saturation=cfg.saturation, 
        hue=cfg.hue)


def HorizontalFlip(cfg):
    return T.RandomHorizontalFlip(p = cfg)


def Normalize(cfg):
    return [T.ToTensor(), T.Normalize(mean=cfg.mean, std=cfg.std)] # T.Normalize() expects tensor


def Pad(cfg):
    return T.Pad(tuple(cfg))


def PadSquare(cfg):
    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp, vp)
            return F.pad(image, padding, 0, 'constant')
        def __repr__(self) -> str:
            return f"{self.__class__.__name__}()"
    return SquarePad()


def Perspective(cfg):
    return T.RandomPerspective(distortion_scale=cfg.distortion_scale, p=cfg.p)


def Resize(cfg):
    return T.Resize((cfg, cfg))


def Rotation(cfg):
    return T.RandomRotation(degrees = cfg)


def ToTensor(cfg):
    return T.ToTensor()
