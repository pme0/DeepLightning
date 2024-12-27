import json

from deeplightning import DATA_REGISTRY
from deeplightning import METRIC_REGISTRY
from deeplightning import MODEL_REGISTRY
from deeplightning import TASK_REGISTRY

import deeplightning.datasets
import deeplightning.metrics
import deeplightning.models
import deeplightning.tasks


if __name__ == "__main__":

    REGISTRIES = [
        DATA_REGISTRY,
        METRIC_REGISTRY,
        MODEL_REGISTRY,
        TASK_REGISTRY
    ]

    ind = 4

    for registry in REGISTRIES:

        print(
            "{} = {}".format(
                registry.registry_type.upper(),
                json.dumps(
                    sorted(registry.get_element_names()), indent=ind,
                ),
            )
        )