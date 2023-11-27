from typing import Tuple, Union, List
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import inspect

from deeplightning import METRIC_REGISTRY
        
        
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
        defaults: dictionary of default lists of metrics for each stage,
            `{"train": ["m1"], "val": ["m2"], "test": ["m2", "m3"]}`.

    Attributes:
        metrics_dict: dictionary of the form `{"train": x, "val": y, "test": z}`
            where `x, y, z` are either "default" (in which case a default set of 
            metrics is used, as defined in the task class) or a list of metric 
            names read from the config.
    """
    def __init__(self, cfg: OmegaConf, defaults: dict) -> None:
        self.metrics_dict = initialise_metrics(cfg=cfg, defaults=defaults)


    def _update_single_metric(self, stage: str, metric_name: str, **kwargs) -> None:
        """
        """
        # Extract update function from the metrics dict
        fn = self.metrics_dict[stage][metric_name].update
        
        # Determine update function's arg names
        fn_arg_names = [p.name for p in inspect.signature(fn).parameters.values()]
        
        # Construct update function's arg dict from `kwargs`
        fn_args = {k: v for k, v in kwargs.items() if k in fn_arg_names}
        
        # Call update function with the approproate args
        fn(**fn_args)


    def update(self, stage, metric_names: Union[str, List[str]] = "all", **kwargs) -> None:
        """Updates metrics using the corresponding `update` function.
        """
        if metric_names == "all":
            metric_names = self.metrics_dict[stage].keys()
        for metric_name in metric_names:
            self._update_single_metric(stage, metric_name, **kwargs)

            


    def compute(self):
        """Computes metrics using the corresponding `compute` function.
        """
        #TODO
        raise NotImplementedError


    def reset(self):
        """Resets metrics using the corresponding `reset` function.
        """
        #TODO
        raise NotImplementedError


# Auxiliary 


def initialise_metrics(cfg: OmegaConf, defaults: dict = None) -> dict:
    metrics_dict = {}
    for stage in ["train", "val", "test"]:
        metrics_dict[stage] = {}
        metrics_list = metrics_defaults(cfg, stage, defaults)
        for metric_name in metrics_list:
            metrics_dict[stage][metric_name] = METRIC_REGISTRY.get_element_instance(
                name=metric_name, cfg=cfg)
    return metrics_dict
    

def metrics_defaults(cfg: OmegaConf, stage: str, defaults: dict = None) -> list:
    if defaults is None or isinstance(cfg.metrics[stage], ListConfig):
        return cfg.metrics[stage]
    elif cfg.metrics[stage] == "default":
        return defaults[stage]
    else:
        raise ValueError