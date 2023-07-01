import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)]
        layers += [nn.BatchNorm2d(out_channels)] if use_batchnorm else []
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)]
        layers += [nn.BatchNorm2d(out_channels)] if use_batchnorm else []
        layers += [nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=True)]
        layers += [nn.BatchNorm2d(out_channels)] if use_batchnorm else []
        layers += [nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsample(x)
        # The following padding is not clear from the paper how it should be 
        # implemented but it is necessary in order to get to right feature map 
        # shapes. Should we pad from right or left, top or bottom?
        x = F.pad(x, (0, 1, 0, 1))
        x = self.layers(x)
        return x


class UNet(nn.Module):
    """UNet Architecture
    
    The authors explain the architecture: ' It consists of a contracting
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
        
    References
    ----------
    'U-Net: Convolutional Networks for Biomedical Image Segmentation', https://arxiv.org/abs/1505.04597
        
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, 64, use_batchnorm)
        self.conv2 = ConvBlock(64, 128, use_batchnorm)
        self.conv3 = ConvBlock(128, 256, use_batchnorm)
        self.conv4 = ConvBlock(256, 512, use_batchnorm)
        self.conv5 = ConvBlock(512, 1024, use_batchnorm)
        self.upsample_conv_5 = UpsampleConvBlock(1024, 512, use_batchnorm)
        self.up_conv5 = ConvBlock(1024, 512, use_batchnorm)
        self.upsample_conv_4 = UpsampleConvBlock(512, 256, use_batchnorm)
        self.up_conv4 = ConvBlock(512, 256, use_batchnorm)
        self.upsample_conv_3 = UpsampleConvBlock(256, 128, use_batchnorm)
        self.up_conv3 = ConvBlock(256, 128, use_batchnorm)
        self.upsample_conv_2 = UpsampleConvBlock(128, 64, use_batchnorm)
        self.up_conv2 = ConvBlock(128, 64, use_batchnorm)
        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = torch.nn.Sigmoid()

    def crop(self, source, target):
        _, _, w_old, h_old = source.size()  # get the original width and height
        _, _, w_new, h_new = target.size()  # get the original width and height
        left = (w_old - w_new) // 2  # calculate the left offset for cropping
        top = (h_old - h_new) // 2  # calculate the top offset for cropping
        return source[:, :, left:left+w_new, top:top+h_new]

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.maxpool(e1)
        e2 = self.conv2(e2)
        e3 = self.maxpool(e2)
        e3 = self.conv3(e3)
        e4 = self.maxpool(e3)
        e4 = self.conv4(e4)
        e5 = self.maxpool(e4)
        x = self.conv5(e5)
        x = self.upsample_conv_5(x)
        x = torch.cat((self.crop(e4, x), x), dim=1)
        x = self.up_conv5(x)
        x = self.upsample_conv_4(x)
        x = torch.cat((self.crop(e3, x), x), dim=1)
        x = self.up_conv4(x)
        x = self.upsample_conv_3(x)
        x = torch.cat((self.crop(e2, x), x), dim=1)
        x = self.up_conv3(x)
        x = self.upsample_conv_2(x)
        x = torch.cat((self.crop(e1, x), x), dim=1)
        x = self.up_conv2(x)
        x = self.last_conv(x)
        x = self.activation(x)
        return x