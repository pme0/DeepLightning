#from lightning.pytorch.loggers import WandbLogger

#from deeplightning.logger.wandb import wandbLogger
"""
from deeplightning.trainer.hooks.ImageClassification_hooks import (
    training_step__ImageClassification,
    training_step_end__ImageClassification,
    on_training_epoch_end__ImageClassification,
    validation_step__ImageClassification,
    validation_step_end__ImageClassification,
    on_validation_epoch_end__ImageClassification,
    test_step__ImageClassification,
    test_step_end__ImageClassification,
    on_test_epoch_end__ImageClassification)
from deeplightning.trainer.hooks.SemanticSegmentation_hooks import (
    training_step__SemanticSegmentation,
    training_step_end__SemanticSegmentation,
    on_training_epoch_end__SemanticSegmentation,
    validation_step__SemanticSegmentation,
    validation_step_end__SemanticSegmentation,
    on_validation_epoch_end__SemanticSegmentation,
    test_step__SemanticSegmentation,
    test_step_end__SemanticSegmentation,
    on_test_epoch_end__SemanticSegmentation)

from deeplightning.trainer.hooks.AudioClassification_hooks import (
    training_step__AudioClassification,
    training_step_end__AudioClassification,
    training_epoch_end__AudioClassification,
    validation_step__AudioClassification,
    validation_step_end__AudioClassification,
    validation_epoch_end__AudioClassification,
    test_step__AudioClassification,
    test_step_end__AudioClassification,
    test_epoch_end__AudioClassification)

from deeplightning.utils.metrics import Metric_Accuracy, Metric_ConfusionMatrix, Metric_PrecisionRecallCurve
"""

from typing import Any, Callable, List, Type, TypeVar
T = TypeVar('T')


__REGISTRIES__ = [
    "tasks",
    "models",
    "datasets",
    "metrics",
]


class Registry:
    """Registers all elements and prevents multiple elements with the same name
    """
    def __init__(self, registry_type: str):
        assert registry_type in __REGISTRIES__
        self.registry_type = registry_type
        self.elements_dict = {}
        

    def register_element(self, name: str = None) -> Callable:
        """Register an element
        """
        def decorator(fn: Callable) -> Callable:
            key = name if name is not None else fn.__name__
            if key in self.elements_dict:
                raise ValueError(
                    f"An entry is already registered under the name '{key}': "
                    f"{self.elements_dict[key]}"
                )
            self.elements_dict[key] = fn
            return fn
        return decorator

    def get_element_reference(self, name: str) -> Type[T]:
        """Get a element reference from its name
        """
        return self.elements_dict[name]
    
    def get_element_instance(self, name: str, **params: Any) -> Callable:
        """Get a element instance from its name and parameters
        """
        return self.get_element_reference(name)(**params)
    
    def get_element_names(self) -> List:
        """Get the names of all registered elements
        """
        return sorted(list(self.elements_dict.keys()))


TASK_REGISTRY = Registry("tasks")
MODEL_REGISTRY = Registry("models")
DATA_REGISTRY = Registry("datasets")
METRICS_REGISTRY = Registry("metrics")



'''
__TaskRegistry__ = [
    # Image
    "ImageClassification",
    "ImageReconstruction",
    "ObjectDetection",
    "SemanticSegmentation",
    # Audio
    "AudioClassification",
]

__HooksRegistry__ = {
    # Image
    "ImageClassification": {
        "training_step": training_step__ImageClassification,
        "training_step_end": training_step_end__ImageClassification,
        "on_training_epoch_end": on_training_epoch_end__ImageClassification,
        "validation_step": validation_step__ImageClassification,
        "validation_step_end": validation_step_end__ImageClassification,
        "on_validation_epoch_end": on_validation_epoch_end__ImageClassification,
        "test_step": test_step__ImageClassification,
        "test_step_end": test_step_end__ImageClassification,
        "on_test_epoch_end": on_test_epoch_end__ImageClassification,
        "LOGGED_METRICS_NAMES": [
            "train_loss", "train_acc", 
            "val_loss", "val_acc", "val_confusion_matrix", "val_precision_recall",
            "test_loss", "test_acc", "test_confusion_matrix", "test_precision_recall",
            "lr"],
    },
    "SemanticSegmentation": {
        "training_step": training_step__SemanticSegmentation,
        "training_step_end": training_step_end__SemanticSegmentation,
        "on_training_epoch_end": on_training_epoch_end__SemanticSegmentation,
        "validation_step": validation_step__SemanticSegmentation,
        "validation_step_end": validation_step_end__SemanticSegmentation,
        "on_validation_epoch_end": on_validation_epoch_end__SemanticSegmentation,
        "test_step": test_step__SemanticSegmentation,
        "test_step_end": test_step_end__SemanticSegmentation,
        "on_test_epoch_end": on_test_epoch_end__SemanticSegmentation,
        "LOGGED_METRICS_NAMES": [
            "train_loss", "train_acc", 
            "val_loss", "val_acc", "val_confusion_matrix", "val_precision_recall",
            "test_loss", "test_acc", "test_confusion_matrix", "test_precision_recall",
            "lr"],
    },
    # Audio
    "AudioClassification": {
        "LOGGED_METRICS_NAMES": [
            "train_loss", "train_acc", 
            "val_loss", "val_acc", "val_confusion_matrix", "val_precision_recall",
            "test_loss", "test_acc", "test_confusion_matrix", "test_precision_recall",
            "lr"],
    },
}


__MetricsRegistry__ = {
    # Image
	"ImageClassification": {
        "Accuracy_train": Metric_Accuracy,
        "Accuracy_val": Metric_Accuracy,
        "Accuracy_test": Metric_Accuracy,
        "ConfusionMatrix_val": Metric_ConfusionMatrix,
        "ConfusionMatrix_test": Metric_ConfusionMatrix,
		"PrecisionRecallCurve_val": Metric_PrecisionRecallCurve,
		"PrecisionRecallCurve_test": Metric_PrecisionRecallCurve,
	},
    "ImageReconstruction": {
        "_": None,
    },
    "ObjectDetection": {
        "_": None,
    },
    "SemanticSegmentation": {
        "Accuracy_train": Metric_Accuracy,
        "Accuracy_val": Metric_Accuracy,
        "Accuracy_test": Metric_Accuracy,
    },
    # Audio
    "AudioClassification":{
        "Accuracy": Metric_Accuracy,
        "ConfusionMatrix": Metric_ConfusionMatrix,
		"PrecisionRecallCurve": Metric_PrecisionRecallCurve,
    },
}



__LoggerRegistry__ = {
    "wandb": WandbLogger,
}
'''