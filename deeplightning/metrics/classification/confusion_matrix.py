from typing import Callable
from omegaconf import OmegaConf
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import wandb

from deeplightning import METRIC_REGISTRY


__all__ = [
	"ConfusionMatrix", 
	"confusion_matrix",
]


class ConfusionMatrix(MulticlassConfusionMatrix):
	"""Confusion Matrix metric class, inheriting from torchmetrics.

	Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
	"""
	def __init__(self, cfg: OmegaConf):
		self.display_name = "confusion_matrix"
		self.logging_methods = ["draw"]

		self.num_classes = cfg.model.network.params.num_classes
		args = {
			"num_classes": self.num_classes,
			"normalize": "true",  # 'true' normalizes over true labels (targets)
		}
		super().__init__(**args)


	def draw(self, 
		phase: str,
		metric_tracker: dict,
		logging_key: str,
		epoch: int,
		max_epochs: int,
	) -> None:
		"""Draw Confusion Matrix as a figure, to be logged as artifact media.

		Args:
			phase: trainer phase, either "train", "val", or "test".
			metric_tracker: dictionary of metrics logged.
			key: name under which metric is logged.
			epoch: current epoch, for labelling (0-indexed).
			max_epochs: number of training epochs.
		"""

		# Compute confusion matrix
		cm = self.compute()
		cm = np.round(100*cm.cpu().numpy()).astype(int)
		assert self.num_classes == cm.shape[0] == cm.shape[1]
		
		# Draw figure
		figure = plt.subplot()
		cbar_args = {
			"label": "Correct predictions (%), normalized by true class"}
		sn.heatmap(
			data = cm, 
			annot=True, fmt="g", square=True, cmap="Blues", 
			vmin=0, vmax=100, cbar_kws=cbar_args)
		plt.title(f"Confusion Matrix [{phase}, epoch {epoch+1}/{max_epochs}]")
		plt.xlabel("Predicted class")
		plt.ylabel("True class")
		plt.close()

		# Save figure
		caption = f"Confusion Matrix [val, epoch {epoch+1}/{max_epochs}]"
		metric_tracker[logging_key] = wandb.Image(figure, caption=caption)


@METRIC_REGISTRY.register_element()	
def confusion_matrix(cfg) -> ConfusionMatrix:
    return ConfusionMatrix(cfg)