from typing import Tuple, Union, List
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

from deeplightning import METRIC_REGISTRY


def init_metrics(cfg, defaults) -> dict:
    metrics_dict = {}
    for subset in ["train", "val", "test"]:
        metrics_dict[subset] = {}
        metrics_list = metrics_filter(cfg, subset, defaults)
        for metric_name in metrics_list:
            metrics_dict[subset][metric_name] = metric_name
    return metrics_dict
    

def metrics_filter(cfg, subset, defaults) -> list:
    if isinstance(cfg.metrics[subset], ListConfig):
        return cfg.metrics[subset]
    elif cfg.metrics[subset] == "default":
        return defaults[subset]
    else:
        raise ValueError
        
        
class Metrics():
    """Class for model evaluation metrics.

    Requires the following structure to be present in the config defining a list
    of metrics to be computed during training/validation/testing:
    ```
    metrics:
        train: Union['default', List]
        val: Union['default', List]
        test: Union['default', List]
    ```

    Args:
        cfg: yaml configuration object
        defaults: dictionary of default lists of metrics for each subset,
            `{"train": ["m1"], "val": ["m1", "m2"], "test": ["m1", "m2"]}`.

    Attributes:
        metrics_dict: dictionary of the form `{"train": x, "val": y, "test": z}`
            where `x, y, z` are either "default" or a list of metric names.
    """
    def __init__(self, cfg: OmegaConf, defaults: dict) -> None:
        self.metrics_dict = init_metrics(cfg=cfg, defaults=defaults)


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
