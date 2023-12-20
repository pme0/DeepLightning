from typing import Any, List, Union
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import inspect
from torch import Tensor

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
        defaults: dictionary of default lists of metrics for each stage, for
            example `{"train": ["m1"], "val": ["m2"], "test": ["m2", "m3"]}`.

    Attributes:
        md: metrics dictionary of the form `{"train": x, "val": y, "test": z}`
            where `x, y, z` are either "default" (in which case a default set of 
            metrics is used, as defined in the task class attribute) or a list 
            of metric names read from the config.
    """
    def __init__(self, cfg: OmegaConf, defaults: dict) -> None:
        self.md = initialise_metrics(cfg=cfg, defaults=defaults)


    def _all_metrics_if_unspecified(self, stage, metric_names) -> List[str]:
        if not metric_names:
            return self.md[stage].keys()
        return metric_names
    

    def _get_logging_methods(self, stage: str, metric_name: str):
        return self.md[stage][metric_name].logging_methods


    def _get_logging_key(self, stage: str, metric_name: str):
        return "{}_{}".format(stage, self.md[stage][metric_name].display_name)


    def _call_metric_method(self, 
        stage: str, 
        method_name: str, 
        metric_name: str,
        **kwargs
    ) -> Union[None, Any]:
        """Call metric method.

        Args:
            stage: trainer stage, one of {"train", "val", "test"}.
            method_name: name of method to call.
            metric_name: name of metric.
        """
        assert stage in ["train", "val", "test"]
        # Merge named args and kwargs
        all_args = {
            "stage": stage, 
            "method_name": method_name, 
            "metric_name": metric_name, 
            **kwargs}
        # Extract method
        fn = getattr(self.md[stage][metric_name], method_name)
        # Determine arg names
        fn_arg_names = [p.name for p in inspect.signature(fn).parameters.values()]
        # Construct arg dict
        fn_args_dict = {k: v for k, v in all_args.items() if k in fn_arg_names}
        # Call method
        return fn(**fn_args_dict)


    def update(self, 
        stage: str,
        metric_names: List[str] = [],
        **kwargs,
    ) -> None:
        """Update metrics accumulators using the corresponding `update` method.
        """
        metric_names = self._all_metrics_if_unspecified(stage, metric_names)
        for metric_name in metric_names:
            args = {
                "stage": stage,
                "method_name": "update",
                "metric_name": metric_name,
                **kwargs}
            self._call_metric_method(**args)


    def compute(self, 
        stage: str, 
        curr_metrics: dict,
        metric_names: List[str] = [],
        reset: bool = False, 
        **kwargs,
    ) -> None:
        """Compute metrics using the corresponding `compute` method.
        """
        metric_names = self._all_metrics_if_unspecified(stage, metric_names)
        
        for metric_name in metric_names:
            logging_methods = self._get_logging_methods(stage, metric_name)
            logging_key = self._get_logging_key(stage, metric_name)
            for method_name in logging_methods:
                args = {
                    "stage": stage,
                    "method_name": method_name,
                    "metric_name": metric_name,
                    "curr_metrics": curr_metrics,
                    "logging_key": logging_key,
                    **kwargs}
                value = self._call_metric_method(**args)
                if value is not None:
                    if isinstance(value, Tensor):
                        value = value.item()
                    curr_metrics.update({logging_key: value})
            
            if reset:
                self._call_metric_method(
                    stage = stage,
                    method_name = "reset",
                    metric_name = metric_name,
                )

    '''
    def reset(self, 
        stage: str, 
        metric_names: List[str] = [],
         **kwargs,
    ) -> None:
        """Reset metrics accumulators using the corresponding `reset` method.
        """
        metric_names = self._all_metrics_if_unspecified(stage, metric_names)
        for metric_name in metric_names:
            self._call_metric_method(
                stage = stage,
                method_name = "reset",
                metric_name = metric_name,
            )
    '''


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