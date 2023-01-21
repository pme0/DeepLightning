from typing import Tuple, Union
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as pltFigure


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Multi-Headed Self-Attention block

        References
        ----------
        ORIGINAL PAPER
            "Attention Is All You Need"
            https://arxiv.org/abs/1706.03762

        Parameters
        ----------
        embed_dim: dimensionality of input and attention feature vectors
        hidden_dim: dimensionality of hidden layer in feed-forward network,
            usually 2-4x larger than `embed_dim`
        num_heads: number of heads in the attention layer
        dropout: dropout probability applied in the linear layer
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]  # `attn` returns tuple (attn_output, attn_output_weights)
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) architecture.

    References
    ----------
    ORIGINAL PAPER
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        https://arxiv.org/abs/2010.11929
    TRAINING TIPS 
        "How to train your ViT? Data, Augmentation, and  Regularization in Vision Transformers"
        https://arxiv.org/abs/2106.10270
        "When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations"
        https://arxiv.org/abs/2106.01548

    Parameters
    ----------
   
    [IMAGE]
    num_channels : number of channels of the input (3 for RGB)
    image_size : image size `dim` for square images `(dim, dim)`, or 
        `(width, height)` for rectangular images
    
    [PATCHING & EMBEDDING]
    patch_size : number of pixels per dimension in each patch; patches are 
        assumed square (`patch_size * patch_size`)
    embed_dim : size of the patch embeddings
    
    [TRANSFORMER]
    hidden_dim : size of the hidden layer in the MLP
    num_heads : number of heads in the Multi-Head Attention block
    num_layers : number of layers in the Transformer
    dropout - probability of dropout in the MLP
    
    [CLASSIFIER]
    num_classes : number of classes in the MLP

    """

    def __init__(self,
        image_size: Union[int, Tuple[int,int]],
        embed_dim: int,
        hidden_dim: int,
        num_channels: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        patch_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_size = patch_size

        # check patchify parameters 
        image_w, image_h = pair(image_size)
        assert (image_w % patch_size) == 0, f"image width ({image_w}) must be divisible by patch size ({patch_size})"
        assert (image_h % patch_size) == 0, f"image height ({image_h}) must be divisible by patch size ({patch_size})"
        num_patches = (image_w // patch_size) * (image_h // patch_size)

        # Layers/Networks
        self.embed = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)))
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))


    def patchify(self, x: torch.Tensor, patch_size: int, flatten_pixels: bool = True) -> torch.Tensor:
        """Convert image into patches.

        The implementation used is equivalent to the one below but is slightly faster though more verbose:
        # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
        # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)

        Parameters
        ----------
        x : image tensor of shape [B, C, H, W]
        patch_size : number of pixels per dimension in each patch; patches are 
            assumed square (`patch_size x patch_size`)
        flatten_pixels : if True, pixels in patches will be returned in flattened
            as a feature vector instead of a 2D grid with 1 or 3 channels.

        """

        B, C, H, W = x.shape
        # H' : number of patches in the height dimension (H // patch_size)
        # W' : number of patches in the width dimension (W // patch_size)
        # pH : patch size (in pixels) in the height dimension
        # pW : patch size (in pixels) in the width dimension
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size) # [B, C, H', pH, W', pW]
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, H', W', pH, pW, C]
        x = x.flatten(1, 2)  # [B, H' * W', pH, pW, C]
        if flatten_pixels:
            x = x.flatten(2, 4)  # [B, H' * W', C * pH * pW]
        return x


    @staticmethod
    def show_patches(x: Union[torch.Tensor, str], resize: Tuple[int,int] = None) -> pltFigure:
        """Plot a grid of image patches.

        Parameters
        ----------
        x : either a path to an image or an image tensor of size [P,H,W,C] where P is
            number of patches (flattened), H is patch height, W is patch width, C is 
            number of channels
        resize : target size if `x` is a path

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


    def forward(self, x):

        # Preprocess input
        x = self.patchify(x, self.patch_size)
        B, T, _ = x.shape
        x = self.embed(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embedding[:, : T + 1]
        x = self.dropout(x)

        # Apply Transforrmer
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out