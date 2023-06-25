import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5

    Parameters
    ----------
    num_classes : number of classes
    """
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x