import torch
import torch.nn as nn


class SimpleGAN(nn.Module):
    def __init__(self, batch_size: int, sample_size: int, image_size: int, feature_dim: int):
        """
        """
        super(SimpleGAN, self).__init__()
        
        self.generator = Generator(
            batch_size = batch_size, 
            sample_size = sample_size, 
            image_size = image_size,
            feature_dim = feature_dim,
        )
        
        self.discriminator = Discriminator(
            image_size = image_size, 
            feature_dim = feature_dim,
        )

    def forward(self):
        pass
    

class Generator(nn.Module):
    def __init__(self, batch_size: int, sample_size: int, image_size: int, feature_dim: int):
        """
        """
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.image_size = image_size
        self.feature_dim = feature_dim

        self.generator = nn.Sequential([
            nn.Linear(sample_size, feature_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(feature_dim, image_size ** 2),
            nn.Sigmoid()
        ])


    def forward(self):
        # generate noise
        noise = torch.randn(self.batch_size, self.sample_size)
        # generator output
        output = self.generator(noise)
        # convert the output to greyscale image (1x28x28)
        generated_images = output.reshape(self.batch_size, 1, self.sample_size, self.sample_size)
        return generated_images


class Discriminator(nn.Module):
    def __init__(self, image_size: int, feature_dim: int):
        """
        """
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.feature_dim = feature_dim

        self.discriminator = nn.Sequential([
            nn.Linear(image_size ** 2, feature_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(feature_dim, 1)
        ])
        

    def forward(self, images: torch.Tensor):
        prediction =  self.discriminator(images.reshape(-1, self.image_size ** 2))
        #loss = F.binary_cross_entropy_with_logits(prediction, targets)
        return prediction

