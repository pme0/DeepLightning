import torch
import torch.nn as nn


class SimpleGAN(nn.Module):
    def __init__(self, batch_size: int, sample_size: int, image_size: int, latent_dim: int):
        """
        """
        super(SimpleGAN, self).__init__()
        
        self.generator = Generator(
            batch_size = batch_size, 
            sample_size = sample_size, 
            image_size = image_size,
            latent_dim = latent_dim,
        )
        
        self.discriminator = Discriminator(
            image_size = image_size, 
            latent_dim = latent_dim,
        )

        


    def forward(self, images):
        return self.discriminator(images)
    

class Generator(nn.Module):
    def __init__(self, batch_size: int, sample_size: int, image_size: int, latent_dim: int):
        """
        """
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.generator = nn.Sequential([
            nn.Linear(sample_size, latent_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(latent_dim, image_size ** 2),
            nn.Sigmoid()
        ])


    def forward(self):
        # generate noise
        noise = torch.randn(self.batch_size, self.sample_size)
        # generate images
        generated_images = self.generator(noise)
        # convert the generated images to greyscale image (1x28x28)
        generated_images = generated_images.reshape(self.batch_size, 1, self.sample_size, self.sample_size)
        return generated_images


class Discriminator(nn.Module):
    def __init__(self, image_size: int, latent_dim: int):
        """
        """
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim

        self.discriminator = nn.Sequential([
            nn.Linear(image_size ** 2, latent_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        ])
        

    def forward(self, images: torch.Tensor):
        predictions = self.discriminator(images.reshape(-1, self.image_size ** 2))
        return predictions

