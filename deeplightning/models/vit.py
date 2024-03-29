from typing import Any, Tuple, Union
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as pltFigure
import omegaconf

from deeplightning import MODEL_REGISTRY


__all__ = [
    "VisionTransformer",
    "vision_transformer",
]


pair = lambda x: x if isinstance(x, tuple) or isinstance(x, omegaconf.listconfig.ListConfig) else (x, x)


class Patchify(nn.Module):
    """Reshape image into patches.

    The implementation used is equivalent to the one below but is slightly 
    faster though more verbose:
    ```Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)```
    (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

    Args:
        x : image tensor of shape [B, C, H, W]
        patch_size : number of pixels per dimension in each patch; patches are 
            assumed square (`patch_size x patch_size`)
        flatten_pixels : if True, pixels in patches will be returned in flattened
            as a feature vector instead of a 2D grid with 1 or 3 channels.
    """
    def __init__(self, patch_size: int, flatten_pixels: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten_pixels = flatten_pixels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # P : patch size (in pixels)
        # H' : number of patches in the height dimension (H // P)
        # W' : number of patches in the width dimension (W // P)
        P = self.patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // P, P, W // P, P) # [B, C, H', P, W', P]
        x = x.permute(0, 2, 4, 3, 5, 1)           # [B, H', W', P, P, C]
        x = x.flatten(1, 2)                       # [B, H' * W', P, P, C]
        if self.flatten_pixels:
            x = x.flatten(2, 4)                   # [B, H' * W', C * P * P]
        return x
    
    @staticmethod
    def show_patches(x: Union[torch.Tensor, str], resize: Tuple[int,int] = None) -> pltFigure:
        """Plot a grid of image patches.

        Args:
            x : either a path to an image or an image tensor of size [P,H,W,C] 
                where P is number of patches (flattened), H is patch height, W is 
                patch width, C is number of channels.
            resize : target size if `x` is a path.
        """
        if isinstance(x, str):
            x = Image.open(x)
            x = T.Resize()(pair(resize))
            x = T.ToTensor()(x)

        P, _, _, _ = x.shape
        rows = int(math.sqrt(P))
        cols = int(math.sqrt(P))
        
        fig = plt.figure(figsize=(4,4))
        gs = fig.add_gridspec(nrows=rows, ncols=cols, hspace=0.1, wspace=0.1)
        axs = gs.subplots()

        p = 0
        for i in range(rows):
            for j in range(cols):
                img = x[p,:,:,:]
                axs[i,j].imshow(img)
                axs[i,j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[i,j].axis('off')
                p += 1


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        
    def forward(self, x: torch.Tensor, num_patches: int) -> torch.Tensor:
        x += self.pos_embedding[:, : num_patches + 1]
        return x
    
    
class ClassEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        return x
    

class LinearEmbedding(nn.Module):
    def __init__(self, num_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.linear_embedding = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return self.linear_embedding(x)
    

class Embedding(nn.Module):
    """Embed patches

    Args:
        num_channels: number of channels in image
        image_size: size of image
        patch_size: size of patches (in pixels)
        embed_dim : size of embedding
    """
    def __init__(self, 
        num_channels: int, 
        image_size: Tuple[int,int], 
        patch_size: int, 
        embed_dim: int
    ):
        super().__init__()
        image_size = pair(image_size)
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.linear_embedding = LinearEmbedding(num_channels, patch_size, embed_dim).linear_embedding
        self.class_embedding = ClassEmbedding(embed_dim)
        self.positional_embedding = PositionalEmbedding(self.num_patches, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.linear_embedding(x)
        x = self.class_embedding(x, batch_size)
        x = self.positional_embedding(x, self.num_patches)
        return x
        
    
class AttentionBlock(nn.Module):
    """Multi-Headed Self-Attention block

    Args:
        embed_dim: dimensionality of input and attention feature vectors
        mlp_dim: dimensionality of hidden layer in feed-forward network,
            usually 2-4x larger than `embed_dim`
        num_heads: number of heads in the attention layer
        dropout: dropout probability applied in the linear layer
        
    """
    def __init__(self, 
        embed_dim: int, 
        mlp_dim: int, 
        num_heads: int, 
        dropout=0.0
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm_1(x)
        x += self.attention(x_norm, x_norm, x_norm)[0]  # `attn` returns tuple (attn_output, attn_output_weights)
        x_norm = self.layer_norm_2(x)
        x += self.linear(x_norm)
        return x


class VisionTransformer(nn.Module):
    """
    Args:
        [input]
            num_channels: number of channels of the input (3 for RGB)
            image_size: image size `dim` for square images `(dim, dim)`, 
                or `(width, height)` for rectangular images
        [patching & embedding]
            embed_dim: size of the patch embeddings
            patch_size: number of pixels per dimension in each patch; 
                patches are assumed square (`patch_size * patch_size`)
        [transformer]
            mlp_dim: size of the hidden layer in the Transformer MLP
            num_heads: number of heads in the Multi-Head Attention block
            num_layers: number of layers in the Transformer
            dropout: probability of dropout in the MLP
        [classifier]
            num_classes: number of classes in the MLP classifier
    """
    def __init__(self,
        image_size: Union[int, Tuple[int,int]],
        embed_dim: int,
        mlp_dim: int,
        num_channels: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        patch_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        image_w, image_h = pair(image_size)
        assert (image_w % patch_size) == 0, \
            f"image width ({image_w}) must be divisible by patch size ({patch_size})"
        assert (image_h % patch_size) == 0, \
            f"image height ({image_h}) must be divisible by patch size ({patch_size})"

        self.patchify = Patchify(patch_size=patch_size)
        self.embedding = Embedding(num_channels, image_size, patch_size, embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, mlp_dim, num_heads, dropout=dropout) 
                for _ in range(num_layers)))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Patching and Embedding
        x = self.patchify(x)
        x = self.embedding(x)
        x = self.dropout(x)
        # Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # Classifier
        cls = x[0]
        x = self.mlp_head(cls)
        return x


@MODEL_REGISTRY.register_element()
def vision_transformer(**kwargs: Any) -> VisionTransformer:
    """Vision Transformer (ViT) architecture

    References:
        Dosovitskiy et al (2020) "An Image is Worth 16x16 Words: Transformers 
        for Image Recognition at Scale".
        <https://arxiv.org/abs/2010.11929>

    Args:
        kwargs: parameters passed to the model class
    """
    return VisionTransformer(**kwargs)