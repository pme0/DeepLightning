from typing import Any
from omegaconf import OmegaConf
from deeplightning.registry import METRICS_REGISTRY, TASK_REGISTRY


class TaskSpecification():
    def __init__(self, cfg: OmegaConf):
        assert cfg.task in TASK_REGISTRY.get_element_names()
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