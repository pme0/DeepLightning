from torchmetrics.classification import MulticlassJaccardIndex

from deeplightning.core.dlconfig import DeepLightningConfig
from deeplightning import METRIC_REGISTRY


class IntersectionOverUnion(MulticlassJaccardIndex):
    """Intersection Over Union (a.k.a. Jaccard Index) metric class, inheriting 
    from torchmetrics.

    Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
    """
    def __init__(self, cfg: DeepLightningConfig):
        self.display_name = "iou"
        self.logging_methods = ["compute"]

        self.num_classes = cfg.task.model.args["num_classes"]
        args = {"num_classes": self.num_classes}
        super().__init__(**args)


@METRIC_REGISTRY.register_element()
def iou(cfg: DeepLightningConfig) -> IntersectionOverUnion:
    return IntersectionOverUnion(cfg)