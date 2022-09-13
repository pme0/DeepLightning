import torch.nn as nn


class SimpleCNN(nn.Module):
    
    def __init__(self, num_classes, num_channels):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16,
                    kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                    kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x