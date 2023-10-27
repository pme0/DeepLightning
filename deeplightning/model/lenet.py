from typing import Any
import torch
import torch.nn as nn

from deeplightning.registry import MODEL_REGISTRY


all = [
    "LeNet5",
    "lenet5",
]

    
class LeNet5(nn.Module):
    """
    Args
        num_classes: number of classes
    """
    def __init__(self, num_classes: int):
        super().__init__()
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
    

@MODEL_REGISTRY.register_model()
def lenet5(**kwargs: Any) -> LeNet5:
    """LeNet5 architecture

    Reference
        Lecun et al (1998) `Gradient-based learning applied to document
        recognition`.
        <https://ieeexplore.ieee.org/abstract/document/726791>

    Args
        **kwargs: parameters passed to the model class
    """
    return LeNet5(**kwargs)