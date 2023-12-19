from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.classification.accuracy import MulticlassAccuracy
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from deeplightning import METRIC_REGISTRY


__all__ = [
	"ClassificationAccuracy",
	"classification_accuracy",
]


class ClassificationAccuracy(MulticlassAccuracy):
    """Classification Accuracy metric class, inheriting from torchmetrics.

    Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
    """
    def __init__(self, cfg: OmegaConf):
        self.display_name = "accuracy"
        self.logging_methods = ["compute"]

        self.num_classes = cfg.model.network.params.num_classes
        args = {
            "num_classes": self.num_classes,
        }
        super().__init__(**args)


@METRIC_REGISTRY.register_element()
def classification_accuracy(cfg: OmegaConf) -> ClassificationAccuracy:
    return ClassificationAccuracy(cfg)