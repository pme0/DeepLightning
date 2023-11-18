from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from deeplightning import METRIC_REGISTRY


__all__ = [
	"ConfusionMatrix", 
	"confusion_matrix",
]


class ConfusionMatrix(MulticlassConfusionMatrix):
	"""Confusion Matrix metric class, inheriting from torchmetrics
	"""
	def __init__(self, cfg: OmegaConf):
		self.num_classes = cfg.model.network.params.num_classes
		args = {
			"num_classes": self.num_classes,
			"normalize": "true",  # 'true' normalizes over true labels (targets)
		}
		super().__init__(**args)


	def draw(self, 
		confusion_matrix: Tensor, 
		stage: str,
		epoch: int,
	) -> Figure:
		"""Draw Confusion Matrix as a figure, to be logged as artifact media

		Args:
			confusion_matrix: confusion matrix values
			stage: data subset {"train", "val", "test"}, for labelling
			epoch: current epoch, for labelling
		"""
		assert self.num_classes == confusion_matrix.shape[0]
		assert self.num_classes == confusion_matrix.shape[1]
		
		# round confusion matrix values
		confusion_matrix = np.round(100*confusion_matrix.cpu().numpy()).astype(int)
		
		# draw figure
		fig = plt.subplot()
		cbar_args = {
			"label": "Correct predictions (%), normalized by true class"}
		sn.heatmap(
			data = confusion_matrix, 
			annot=True, fmt="g", square=True, cmap="Blues", 
			vmin=0, vmax=100, cbar_kws=cbar_args)
		plt.title(f"Confusion Matrix [{stage}, epoch {epoch}]")
		plt.xlabel("Predicted class")
		plt.ylabel("True class")
		plt.close()
		return fig


@METRIC_REGISTRY.register_element()	
def confusion_matrix(cfg) -> ConfusionMatrix:
    return ConfusionMatrix(cfg)