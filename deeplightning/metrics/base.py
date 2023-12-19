from typing import List, Literal, Tuple, Union
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
        defaults: dictionary of default lists of metrics for each stage,
            `{"train": ["m1"], "val": ["m2"], "test": ["m2", "m3"]}`.

    Attributes:
        md: metrics dictionary of the form `{"train": x, "val": y, "test": z}`
            where `x, y, z` are either "default" (in which case a default set of 
            metrics is used, as defined in the task class) or a list of metric 
            names read from the config.
    """
    def __init__(self, cfg: OmegaConf, defaults: dict) -> None:
        self.md = initialise_metrics(cfg=cfg, defaults=defaults)


    def _all_metrics_if_unspecified(self, stage, metric_names) -> List[str]:
        if not metric_names:
            return self.md[stage].keys()
        return metric_names


    def _call_metric_method(self, method_name: str, stage: str, metric_name: str, **kwargs) -> None:
        """Call metric method.

        Args:
            method_name: name of method to call, one of {"update", "compute",
                "reset", "draw"}.
            stage: trainer stage, one of {"train", "val", "test"}.
            metric_name: name of metric.
            **kwargs: keyword arguments to be passed to the metric method 
                function.
        """
        # Merge named args and kwargs
        # TODO: build dict in update()/compute() and receive only **kwargs in `_call_metric_method`
        all_args = {"method_name": method_name, "stage": stage, "metric_name": metric_name, **kwargs}
        # Extract method
        fn = getattr(self.md[stage][metric_name], method_name)
        # Determine arg names
        fn_arg_names = [p.name for p in inspect.signature(fn).parameters.values()]
        # Construct arg dict
        fn_args_dict = {k: v for k, v in all_args.items() if k in fn_arg_names}
        # Call method
        #print(method_name, stage, metric_name, fn_arg_names, fn_args_dict.keys())
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
            update_args_dict = {
                "stage": stage,
                "metric_name": metric_name,
                "method_name": "update",
                **kwargs,
            }
            self._call_metric_method(**update_args_dict)


    def compute(self, 
        existing_metrics: dict,
        stage: str, 
        metric_names: List[str] = [],
        reset: bool = False, 
        **kwargs,
    ) -> None:
        """Compute metrics using the corresponding `compute` method.

        Currently, `draw` and `reset` methods are called indirectly via this 
        `compute` method, though they can be called directly if necessary.
        """
        metric_names = self._all_metrics_if_unspecified(stage, metric_names)
        
        for metric_name in metric_names:
            
            logging_methods = self.md[stage][metric_name].logging_methods
            key = "{}_{}".format(stage, self.md[stage][metric_name].display_name)

            for method_name in logging_methods:
                
                args_dict = {
                    "metric_name": metric_name,
                    "method_name": method_name,
                    "stage": stage,
                    "existing_metrics": existing_metrics,
                    "key": key,
                    **kwargs,
                }

                value = self._call_metric_method(**args_dict)
                
                if value is not None:
                    if isinstance(value, Tensor):
                        value = value.item()
                    existing_metrics.update({key: value})
            
            if reset:
                self._call_metric_method(
                    method_name = "reset",
                    stage = stage,
                    metric_name = metric_name,
                )


    def reset(self, 
        stage: str, 
        metric_names: List[str] = [],
         **kwargs,
    ) -> None:
        """Reset metrics accumulators using the corresponding `reset` method.
        """
        metric_names = self._all_metrics_if_unspecified(stage, metric_names)
        for name in metric_names:
            self._call_metric_method(
                stage = stage,
                metric_name = name,
                method_name = "reset")


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