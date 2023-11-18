from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.classification.precision_recall_curve import MulticlassPrecisionRecallCurve
from matplotlib.figure import Figure
from matplotlib.pyplot import pyplot as plt

from deeplightning import METRIC_REGISTRY


__all__ = [
	"PrecisionRecallCurve", 
	"precision_recall_curve",
]


class PrecisionRecallCurve(MulticlassPrecisionRecallCurve):
	"""Precision-Recall metric class, inheriting from torchmetrics
	"""
	def __init__(self, cfg: OmegaConf):
		self.num_classes = cfg.model.network.params.num_classes
		args = {
			"num_classes": self.num_classes,
		}
		super().__init__(**args)

	def draw(self,
		precision: Tensor,
		recall: Tensor,
		thresholds: Tensor,
		stage: str,
		epoch: int
	) -> Figure:
		"""Draw Precision-Recall Curve as a figure, to be logged as artifact media

		Args:
			precision: precisions values
			recall: recalls values
			thresholds: threshold values
			stage: data subset {"train", "val", "test"}, for labelling
			epoch: current epoch, for labelling
		"""
		assert self.num_classes == len(precision)
		assert self.num_classes == len(recall)
		
		# draw figure
		fig = plt.figure()
		for i in range(self.num_classes):
			plt.plot(recall[i].cpu(), precision[i].cpu(), label=i)
		plt.title(f"Precision-Recall Curve [{stage}, epoch {epoch}]")
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		if self.num_classes <= 10:
			plt.legend(loc="lower left", title="class", fontsize='small')
		plt.close()
		return fig
	

@METRIC_REGISTRY.register_element()
def precision_recall_curve(cfg) -> PrecisionRecallCurve:
    return PrecisionRecallCurve(cfg)