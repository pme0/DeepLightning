from torchmetrics.classification import MulticlassAUROC

from deeplightning.core.dlconfig import DeepLightningConfig
from deeplightning import METRIC_REGISTRY


class AUROC(MulticlassAUROC):
    """Area Under ROC metric class, inheriting from torchmetrics.

    Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
    """
    def __init__(self, cfg: DeepLightningConfig):
        self.display_name = "auroc"
        self.logging_methods = ["compute"]

        self.num_classes = cfg.task.model.args.num_classes
        args = {"num_classes": self.num_classes}
        super().__init__(**args)


@METRIC_REGISTRY.register_element()
def auroc(cfg: DeepLightningConfig) -> AUROC:
    return AUROC(cfg)