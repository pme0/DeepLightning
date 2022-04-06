from typing import Tuple, List, Union
import torch
from torch import Tensor
from torchmetrics.functional import accuracy, mean_squared_error


def metric_accuracy(logits: Tensor, y: Tensor) -> Tensor:
    preds = torch.argmax(logits, dim=1)
    return accuracy(preds, y)

def metric_mse(preds: Tensor, target: Tensor) -> Tensor:
    return mean_squared_error(preds, target, squared = True)

def metric_rmse(preds: Tensor, target: Tensor) -> Tensor:
    return mean_squared_error(preds, target, squared = False)
