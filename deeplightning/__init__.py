from deeplightning.registry import Registry


TASK_REGISTRY = Registry("tasks")
MODEL_REGISTRY = Registry("models")
DATA_REGISTRY = Registry("datasets")
METRIC_REGISTRY = Registry("metrics")