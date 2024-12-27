from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision.transforms import functional as F
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from deeplightning.core.dlconfig import DeepLightningConfig
from deeplightning import METRIC_REGISTRY


class SemanticSegmentationMaskContour():
    """Classification Accuracy metric class, inheriting from torchmetrics.

    Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
    """
    def __init__(self, cfg: DeepLightningConfig):
        self.display_name = "segmentation_contour"
        self.logging_methods = ["draw"]

        self.colors = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
        }

    
    def draw(self, preds, masks, images):
        preds = torch.argmax(preds[:4,], dim=1)


@METRIC_REGISTRY.register_element()
def segmentation_mask_contour(cfg: DeepLightningConfig) -> SemanticSegmentationMaskContour:
    return SemanticSegmentationMaskContour(cfg)