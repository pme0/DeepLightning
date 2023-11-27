from typing import Any, Callable
import torch
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

from deeplightning import MODEL_REGISTRY


all = [
    "MLPMixer",
    "mlp_mixer"
]

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )
    

class MLPMixer(nn.Module):
    def __init__(self, 
        image_size: int, 
        num_channels: int, 
        patch_size: int, 
        dim: int, 
        depth: int, 
        num_classes: int, 
        expansion_factor: int = 4, 
        expansion_factor_token: float = 0.5, 
        dropout: float = 0.0,
    ):
        super().__init__()
        # check patchify parameters 
        image_h, image_w = pair(image_size)
        assert (image_w % patch_size) == 0, f"image width ({image_w}) must be divisible by patch size ({patch_size})"
        assert (image_h % patch_size) == 0, f"image height ({image_h}) must be divisible by patch size ({patch_size})"
        
        num_patches = (image_w // patch_size) * (image_h // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.layers = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * num_channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        self.layers(x)


@MODEL_REGISTRY.register_element()
def mlp_mixer(**kwargs: Any) -> MLPMixer:
    """MLPMixer architecture

    Reference
        Tolstikhin et al (2021) `MLP-Mixer: An all-MLP Architecture for Vision`.
        <https://arxiv.org/abs/2105.01601>
        <https://github.com/lucidrains/mlp-mixer-pytorch>

    Args
        **kwargs: parameters passed to the model class
    """
    return MLPMixer(**kwargs)


