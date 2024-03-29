from typing import Any, Union
from math import log2, sqrt
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F

from deeplightning import MODEL_REGISTRY


__all__ = [
    "DiscreteVAE",
    "discrete_vae",
]


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    """
    Args
        image_size: 
        num_tokens: 
        codebook_dim: 
        num_layers: 
        num_resnet_blocks: 
        hidden_dim: 
        channels: 
        smooth_l1_loss: 
        temperature: 
        straight_through: 
        kl_div_loss_weight: 
        normalization: 
    """
    def __init__(
        self,
        image_size: int = 256,
        num_tokens: int = 512,
        codebook_dim: int = 512,
        num_layers: int = 3,
        num_resnet_blocks: int = 0,
        hidden_dim: int = 64,
        channels: int = 3,
        smooth_l1_loss: bool = False,
        temperature: float = 0.9,
        straight_through: bool = False,
        kl_div_loss_weight: float = 0.0,
        normalization = None #|TODO adapt to allow factor 1 (bw images) or 3 (color) images
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResidualBlock(dec_chans[1]))
            enc_layers.append(ResidualBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.smooth_l1_loss = smooth_l1_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization
    
    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(self, img, return_logits = False, temp = None):
        assert img.shape[-1] == self.image_size \
            and img.shape[-2] == self.image_size, \
                f'input must have the correct image size {self.image_size}'

        img = self.norm(img)
        logits = self.encoder(img)

        if return_logits:
            return logits

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        recon = self.decoder(sampled)

        return recon, 


@MODEL_REGISTRY.register_element()
def discrete_vae(**kwargs) -> DiscreteVAE:
    """Discrete (a.k.a. Vector-Quantized) VAE architecture

    References:
        van den Oord et al (2017) "Neural Discrete Representation Learning".
        <https://arxiv.org/abs/1711.00937>

        Ramesh et al (2021) "Zero-Shot Text-to-Image Generation".
        <https://arxiv.org/abs/2102.12092>


    Args:
        kwargs: parameters passed to the model class
    """
    return DiscreteVAE(**kwargs)