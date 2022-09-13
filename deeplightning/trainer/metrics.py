from typing import Tuple, List, Union
from omegaconf import OmegaConf
import torch
from torch import Tensor
import torchmetrics
from torchmetrics.functional import accuracy, mean_squared_error
import seaborn as sn
import numpy as np
import pandas as pd
from matplotlib.figure import Figure as pltFigure
from matplotlib import pyplot as plt


class MetricsConfusionMatrix(torchmetrics.ConfusionMatrix):
    """
    """
    def __init__(self, cfg: OmegaConf):
        self.num_classes = cfg.model.network.params.num_classes
        args = {
            "num_classes": self.num_classes,
            "normalize": "true",
        }
        super().__init__(**args)

    def draw(self, cm: Tensor) -> pltFigure:
        cm = np.round(100*cm.cpu().numpy()).astype(int)
        df_cm = pd.DataFrame(cm, range(self.num_classes), range(self.num_classes))
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        #ax.set_title("Confusion Matrix")
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, cmap="YlGnBu", annot=True, fmt="d", annot_kws={"size": 16}) # font size
        return fig
    
def metric_accuracy(logits: Tensor, y: Tensor) -> Tensor:
    preds = torch.argmax(logits, dim=1)
    return accuracy(preds, y)

def metric_mse(preds: Tensor, target: Tensor) -> Tensor:
    return mean_squared_error(preds, target, squared = True)

def metric_rmse(preds: Tensor, target: Tensor) -> Tensor:
    return mean_squared_error(preds, target, squared = False)
