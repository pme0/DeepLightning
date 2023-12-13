from typing import Any
import torch
import torch.nn as nn

from deeplightning import MODEL_REGISTRY


__all__ = [
    "DCGAN",
    "dcgan",
    "CDCGAN",
    "cdcgan",
]


def initialize_weights(self) -> None:
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

class DCGAN_Generator(nn.Module):
    """DCGAN Generator
    
    Args
        num_channels: number of channels
        latent_dim: size of z latent vector (i.e. size of generator input)
    """
    def __init__(self, num_channels: int, latent_dim: int):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            # input is `z` going into a convolution of size [batch, latent_dim, 1, 1]
            # conv1
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=128, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # conv2
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # conv3
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # conv4
            nn.ConvTranspose2d(in_channels=32, out_channels=num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        """ Parametrize layer creation:
        last_channels = 32
        channels = [latent_dim] + [last_channels * (2 ** i) for i in range(num_layers-1)][::-1] + [num_channels]
        layers = []
        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=4, stride=2, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU() if i < num_layers-1 else nn.Tanh())
        """

    def forward(self, x):
        return self.generator(x)

    def tensor_size_debugger(self):
        x = torch.ones((1, self.latent_dim, 1, 1))
        for m in self.generator.model.children():
            x = m(x)
            print(m, x.shape)


class DCGAN_Discriminator(nn.Module):
    """DCGAN Discriminator
    
    Args
        num_channels: number of channels
        num_features_d: size of feature maps in discriminator
    """
    def __init__(self, num_channels: int, alpha: float = 0.2):
        super().__init__()
        self.main = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(alpha, inplace=True),
            # ........
        )

    def forward(self, input):
        return self.main(input)

    def tensor_size_debugger(self):
        x = torch.ones(())
        for m in self.discriminator.model.children():
            x = m(x)
            print(m, x.shape)


class DCGAN(nn.Module):
    def __init__(self, batch_size: int, sample_size: int, image_size: int, latent_dim: int):
        super().__init__()
        self.generator = DCGAN_Generator(
            batch_size = batch_size, 
            sample_size = sample_size, 
            image_size = image_size,
            latent_dim = latent_dim,
        )
        self.discriminator = DCGAN_Discriminator(
            image_size = image_size, 
            latent_dim = latent_dim,
        )

    def forward(self, x):
        return None
    

class CDCGAN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.generator = None
        self.discriminator = None

    def forward(self, x):
        return None
    

@MODEL_REGISTRY.register_element()
def dcgan(**kwargs: Any) -> DCGAN:
    """Deep Convolutional Generative Adversarial Network architecture

    References:
        Radford et al (2015) "Unsupervised Representation Learning with 
        Deep Convolutional Generative Adversarial Networks".
        <https://arxiv.org/abs/1511.06434>

    Args:
        kwargs: parameters passed to the model class
    """
    return DCGAN(**kwargs)


@MODEL_REGISTRY.register_element()
def cdcgan(**kwargs: Any) -> CDCGAN:
    """Conditional Deep Convolutional Generative Adversarial Network architecture

    Reference:s
        ?

    Args:
        kwargs: parameters passed to the model class
    """
    return CDCGAN(**kwargs)