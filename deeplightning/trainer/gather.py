from typing import Tuple, List, Union
import torch


def gather_on_step(
    step_outputs: Union[dict, List[dict]], 
    metrics: List[str], 
    average: bool
    ) -> dict:
    """ Aggregate metrics across devices at step_end.

    Arguments:
        :step_outputs: metrics dictionary in single-device 
            training, or list of metrics dictionaries in 
            multi-device training (one element per device).
        :metrics:  metrics to aggregate.
        :average: whether to average (True) or just 
            sum (False) the aggregated values.

    Returns:
        A dictionary with keys `metrics` and values corresponding 
            to the aggregated value for each metric.
    """
    output = {}
    for metric in metrics:
        if isinstance(step_outputs, list):
            agg = [step_outputs[i][metric] for i in range(len(step_outputs))]
        else:
            agg = step_outputs[metric]
        output[metric] = torch.sum(agg).item() / (len(agg) if average is True else 1.0)
    return output


def gather_on_epoch(
    epoch_outputs: Union[dict, List[dict]], 
    metrics: List[str], 
    average: bool
    ) -> dict:
    """ Aggregate metrics across steps at epoch_end.

    Arguments:
        :epoch_outputs: metrics dictionary in single-device 
            training, or list of metrics dictionaries in 
            multi-device training (one element per device).
        :metrics: metrics to aggregate.
        :average: whether to average (True) or just
            sum (False) the aggregated values.

    Returns:
        A dictionary with keys `metrics` and values corresponding 
        to the aggregated value for each metric.
    """
    output = {}
    for metric in metrics:
        if isinstance(epoch_outputs, list):
            agg = [epoch_outputs[i][metric] for i in range(len(epoch_outputs))]
        else:
            agg = epoch_outputs[metric]
        output[metric] =  sum(agg) / (len(agg) if average is True else 1.0)
    return output