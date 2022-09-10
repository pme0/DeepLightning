from torchvision import transforms as T
import torchvision.transforms.functional as F
import numpy as np


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