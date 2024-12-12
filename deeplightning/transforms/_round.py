import torch


class RoundToIntegerTransform():
    def __call__(self, x):
        return torch.round(x).long()


def RoundToInteger():
    """Converts tensor to integer by rounding.
    
    This is useful after resizing segmentation masks as the interpolation 
    method used in the resizing transform introduces non-integer values.
    """
    return RoundToIntegerTransform()