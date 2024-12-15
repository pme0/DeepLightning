import torch


class RoundToInteger_class(torch.nn.Module):
        """Converts tensor to integer by rounding.
        
        This is useful after resizing segmentation masks as the interpolation method
        used in the resizing transform introduces non-integer values.
        """
        def __init__(self):
            super().__init__()

        def __call__(self, x):
            return torch.round(x).long()
        
        def __repr__(self):
            return "RoundToInteger()"
        

def RoundToInteger():
    return RoundToInteger_class()