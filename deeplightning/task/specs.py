from typing import Any
from omegaconf import OmegaConf
from deeplightning.registry import METRICS_REGISTRY, TASK_REGISTRY


__TASKS__ = [
    "ImageClassification",
    "SemanticSegmentation",
]


#|TODO register metrics with decorator
__METRICS__ = {"classification_accuracy": None}


class TaskSpecification():
    def __init__(self, cfg: OmegaConf):
        assert cfg.task in __TASKS__
        self.task = cfg.task


class ImageClassificationTask(TaskSpecification):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.metrics = [
            "classification_accuracy",
        ]
        for m in self.metrics:
            assert m in METRICS_REGISTRY.get_element_names()


def ImageClassificationSpec(cfg: OmegaConf) -> ImageClassificationTask:
    return ImageClassificationTask(cfg)