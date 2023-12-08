from typing import Callable
from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from deeplightning import METRIC_REGISTRY


__all__ = [
	"ConfusionMatrix", 
	"confusion_matrix",
]


class ConfusionMatrix(MulticlassConfusionMatrix):
	"""Confusion Matrix metric class, inheriting from torchmetrics.

	Attributes:
		display_name: 
		logging_methods: 
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
		stage: str,
		metric_name: str,
		compute_fn: Callable,
		epoch: int,
		max_epochs: int,
	) -> Figure:
		"""Draw Confusion Matrix as a figure, to be logged as artifact media.

		Args:
			stage: trainer stage, one of {"train", "val", "test"}.
			metric_name: name of metric.
			compute_fn: the metric's compute function.
			epoch: current epoch, for labelling (0-indexed).
			max_epochs: number of training epochs.
		"""

		# compute and round confusion matrix
		confusion_matrix = compute_fn()
		confusion_matrix = np.round(100*confusion_matrix.cpu().numpy()).astype(int)
		assert self.num_classes == confusion_matrix.shape[0]
		assert self.num_classes == confusion_matrix.shape[1]
		
		# draw figure
		fig = plt.subplot()
		cbar_args = {
			"label": "Correct predictions (%), normalized by true class"}
		sn.heatmap(
			data = confusion_matrix, 
			annot=True, fmt="g", square=True, cmap="Blues", 
			vmin=0, vmax=100, cbar_kws=cbar_args)
		plt.title(f"Confusion Matrix [{stage}, epoch {epoch+1}/{max_epochs}]")
		plt.xlabel("Predicted class")
		plt.ylabel("True class")
		plt.close()
		return fig


@METRIC_REGISTRY.register_element()	
def confusion_matrix(cfg) -> ConfusionMatrix:
    return ConfusionMatrix(cfg)