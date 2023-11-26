from typing import Tuple, Union, List
from omegaconf import OmegaConf

from deeplightning import METRIC_REGISTRY


class Metrics():
    """Class for model evaluation metrics.

    Requires the following structure to be present in the config: 
    ```
    metrics:
        train: Union['default', List]
        val: Union['default', List]
        test: Union['default', List]
    ```
    defining a list of metrics to be computed during training/validation/testing.
    """
    def __init__(self, cfg: OmegaConf, defaults: dict) -> None:
        # intialise metrics dictionary
        self.metrics_dict = {}
        for subset in cfg.metrics:
            self.metrics_dict[subset] = {}
            metrics_list = cfg.metrics[subset]
            if metrics_list == "default":
                metrics_list = defaults[subset]
            for m in metrics_list:
                self.metrics_dict[subset][m] = METRIC_REGISTRY.get_element_instance(
                    name=m, cfg=cfg)


    def update(self, subset, metric_names: Union[str, List[str]] = "all"):
        #TODO
        raise NotImplementedError
        if metric_names == "all":
            metric_names = self.metrics_dict[subset].keys()
        for metric_name in metric_names:
            self.metrics_dict[subset][metric_name].update(...)


    def compute(self):
        #TODO
        raise NotImplementedError


    def reset(self):
        #TODO
        raise NotImplementedError
