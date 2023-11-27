from typing import Any
import torch
import torch.nn as nn

from deeplightning import MODEL_REGISTRY


__all__ = [
    "ConvMixer",
    "conv_mixer",
]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, num_classes=1000, num_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


@MODEL_REGISTRY.register_element()
def conv_mixer(**kwargs) -> ConvMixer:
    """ConvMixer architecture

    Reference
        Trockman et al (2022) `Patches Are All You Need?`.
        <https://arxiv.org/abs/2201.09792>
        <https://github.com/locuslab/convmixer>

    Args
        **kwargs: parameters passed to the model class
    """
    return ConvMixer(**kwargs)