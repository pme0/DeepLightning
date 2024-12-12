from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.classification.precision_recall_curve import MulticlassPrecisionRecallCurve
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import wandb

from deeplightning import METRIC_REGISTRY


__all__ = [
	"PrecisionRecallCurve", 
	"precision_recall_curve",
]


class PrecisionRecallCurve(MulticlassPrecisionRecallCurve):
	"""Precision-Recall metric class, inheriting from torchmetrics.

	Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
	"""
	def __init__(self, cfg: OmegaConf):
		self.display_name = "precision_recall"
		self.logging_methods = ["draw"]

		self.num_classes = cfg.task.model.args.num_classes
		args = {"num_classes": self.num_classes}
		super().__init__(**args)

	def draw(self,
		phase: str,
		metric_tracker: dict,
		logging_key: str,
		epoch: int,
		max_epochs: int,
	) -> None:
		"""Draw Precision-Recall Curve as a figure, to be logged as artifact media

		Args:
			phase: trainer phase, either "train", "val", or "test".
			metric_tracker: dictionary of metrics logged.
			key: name under which metric is logged.
			epoch: current epoch, for labelling.
			max_epochs: number of training epochs.
		"""

		# Compute precision and recall
		precision, recall, thresholds = self.compute()
		assert self.num_classes == len(precision)
		assert self.num_classes == len(recall)
		
		# Draw figure
		figure = plt.figure()
		for i in range(self.num_classes):
			plt.plot(recall[i].cpu(), precision[i].cpu(), label=i)
		plt.title(f"Precision-Recall Curve [{phase}, epoch {epoch+1}/{max_epochs}]")
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		if self.num_classes <= 10:
			plt.legend(loc="lower left", title="class", fontsize='small')
		plt.close()

		# Save figure
		caption = f"Precision-Recall Curve [val, epoch {epoch+1}/{max_epochs}]"
		metric_tracker[logging_key] = wandb.Image(figure, caption=caption)
			

@METRIC_REGISTRY.register_element()
def precision_recall_curve(cfg) -> PrecisionRecallCurve:
    return PrecisionRecallCurve(cfg)