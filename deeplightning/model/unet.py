from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


from deeplightning import MODEL_REGISTRY


all = [
    "UNet",
    "unet",
]


class DoubleConv(nn.Module):
    """downsampling with two conv3x3-bn-relu blocks"""
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(DoubleConv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class DownSampleConvBlock(nn.Module):
    """downsample with maxpool followed by doubleconv block"""
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(DownSampleConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.layers(x)


class UpConv(nn.Module):
    """upsampling with conv2x2-relu block"""
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(UpConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

    
class UpsampleConvBlock(nn.Module):
    """upsampling with upsample-upconv-doubleconv block"""
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.upconv = UpConv(in_channels, out_channels)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def crop(self, source, target):
        """crop source tensor to target tensor's shape"""
        _, _, w_old, h_old = source.size()  # get the original width and height
        _, _, w_new, h_new = target.size()  # get the original width and height
        left = (w_old - w_new) // 2  # calculate the left offset for cropping
        top = (h_old - h_new) // 2  # calculate the top offset for cropping
        return source[:, :, left:left+w_new, top:top+h_new]

    def forward(self, x, y):
        """
        x : the input feature map from the expansive path
        y : the corresponding feature map from the contracting path
        """
        cropped_y = self.crop(y, x)
        x = self.upsample(x)
        # The following padding is not clear from the paper how it should be 
        # implemented but it is necessary in order to get the right feature map 
        # shapes. Should we pad from right or left, top or bottom?
        x = F.pad(x, (0, 1, 0, 1))
        x = self.upconv(x)
        cropped_y = self.crop(y, x)
        x = torch.cat((cropped_y, x), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    Args
        in_channels: number of input channels (image channels)
        out_channels: number of output channels (number of classes)
        use_batchnorm: whether to use batch normalization
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = False):
        super(UNet, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=0)
        self.down1 = DownSampleConvBlock(64, 128, use_batchnorm)
        self.down2 = DownSampleConvBlock(128, 256, use_batchnorm)
        self.down3 = DownSampleConvBlock(256, 512, use_batchnorm)
        self.down4 = DownSampleConvBlock(512, 1024, use_batchnorm)
        self.up4 = UpsampleConvBlock(1024, 512, use_batchnorm)
        self.up3 = UpsampleConvBlock(512, 256, use_batchnorm)
        self.up2 = UpsampleConvBlock(256, 128, use_batchnorm)
        self.up1 = UpsampleConvBlock(128, 64, use_batchnorm)
        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.last_conv(x)
        x = self.activation(x)
        return x
    

@MODEL_REGISTRY.register_element()
def unet(**kwargs: Any) -> UNet:
    """UNet architecture

    The authors explain the architecture: 'It consists of a contracting
    path (left side) and an expansive path (right side). The contracting path follows
    the typical architecture of a convolutional network. It consists of the repeated
    application of two 3x3 convolutions (unpadded convolutions), each followed by
    a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
    for downsampling. At each downsampling step we double the number of feature
    channels. Every step in the expansive path consists of an upsampling of the
    feature map followed by a 2x2 convolution (“up-convolution”) that halves the
    number of feature channels, a concatenation with the correspondingly cropped
    feature map from the contracting path, and two 3x3 convolutions, each followed 
    by a ReLU. The cropping is necessary due to the loss of border pixels in
    every convolution. At the final layer a 1x1 convolution is used to map each 
    64-component feature vector to the desired number of classes.'

    Reference
        Ronneberger et al (2015) `U-Net: Convolutional Networks for Biomedical
        Image Segmentation`.
        <https://arxiv.org/abs/1505.04597>

    Args
        **kwargs: parameters passed to the model class
    """
    return UNet(**kwargs)