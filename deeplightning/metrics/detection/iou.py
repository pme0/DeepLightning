from omegaconf import OmegaConf
from torchmetrics.classification import MulticlassJaccardIndex

from deeplightning import METRIC_REGISTRY


__all__ = [
    "IntersectionOverUnion",
    "iou",
]


class IntersectionOverUnion(MulticlassJaccardIndex):
    """Intersection Over Union (a.k.a. Jaccard Index) metric class, inheriting 
    from torchmetrics.

    Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
    """
    def __init__(self, cfg: OmegaConf):
        self.display_name = "iou"
        self.logging_methods = ["compute"]

        self.num_classes = cfg.model.network.params.num_classes
        args = {
            "num_classes": self.num_classes,
        }
        super().__init__(**args)


@METRIC_REGISTRY.register_element()
def iou(cfg: OmegaConf) -> IntersectionOverUnion:
    return IntersectionOverUnion(cfg)