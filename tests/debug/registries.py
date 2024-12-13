from deeplightning.registry import __REGISTRIES__

from deeplightning import TASK_REGISTRY
from deeplightning import MODEL_REGISTRY
from deeplightning import METRIC_REGISTRY
from deeplightning import DATA_REGISTRY

import deeplightning.tasks
import deeplightning.models
import deeplightning.metrics
import deeplightning.datasets

if __name__ == "__main__":

    print(f"TASK_REGISTRY = {sorted(TASK_REGISTRY.get_element_names())}")
    print(f"MODEL_REGISTRY = {sorted(MODEL_REGISTRY.get_element_names())}")
    print(f"METRIC_REGISTRY = {sorted(METRIC_REGISTRY.get_element_names())}")
    print(f"DATA_REGISTRY = {sorted(DATA_REGISTRY.get_element_names())}")
    