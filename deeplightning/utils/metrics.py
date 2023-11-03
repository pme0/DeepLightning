from bdb import effective
from typing import Tuple, List, Union
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
from torchmetrics.classification.precision_recall_curve import MulticlassPrecisionRecallCurve
from torchmetrics.functional.classification.accuracy import accuracy
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from deeplightning import METRIC_REGISTRY
#from deeplightning.utils.metrics import Metric_Accuracy, Metric_ConfusionMatrix, Metric_PrecisionRecallCurve


__all__ = [
	"ClassificationAccuracy", "classification_accuracy",
	"PrecisionRecallCurve", "precision_recall_curve",
	"ConfusionMatrix", "confusion_matrix",
]


'''
__MetricsRegistry__ = {
    # Image
	"ImageClassification": {
        "Accuracy_train": Metric_Accuracy,
        "Accuracy_val": Metric_Accuracy,
        "Accuracy_test": Metric_Accuracy,
        "ConfusionMatrix_val": Metric_ConfusionMatrix,
        "ConfusionMatrix_test": Metric_ConfusionMatrix,
		"PrecisionRecallCurve_val": Metric_PrecisionRecallCurve,
		"PrecisionRecallCurve_test": Metric_PrecisionRecallCurve,
	},
    "ImageReconstruction": {
        "_": None,
    },
    "ObjectDetection": {
        "_": None,
    },
    "SemanticSegmentation": {
        "Accuracy_train": Metric_Accuracy,
        "Accuracy_val": Metric_Accuracy,
        "Accuracy_test": Metric_Accuracy,
    },
    # Audio
    "AudioClassification":{
        "Accuracy": Metric_Accuracy,
        "ConfusionMatrix": Metric_ConfusionMatrix,
		"PrecisionRecallCurve": Metric_PrecisionRecallCurve,
    },
}
'''


class ClassificationAccuracy(MulticlassAccuracy):
    """Classification Accuracy metric, inheriting from torchmetrics
    """
    def __init__(self, cfg: OmegaConf):
        self.num_classes = cfg.model.network.params.num_classes
        args = {
            "num_classes": self.num_classes,
        }
        super().__init__(**args)


@METRIC_REGISTRY.register_element()
def classification_accuracy(cfg) -> ClassificationAccuracy:
    return ClassificationAccuracy(cfg)


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



def metric_accuracy(logits: Tensor, target: Tensor, task: str, num_classes: int) -> Tensor:
	preds = torch.argmax(logits, dim=1)
	return accuracy(preds=preds, target=target, task=task, num_classes=num_classes)


def metric_mse(preds: Tensor, target: Tensor) -> Tensor:
	return mean_squared_error(preds, target, squared = True)
	

def metric_rmse(preds: Tensor, target: Tensor) -> Tensor:
	return mean_squared_error(preds, target, squared = False)


def IOU(box1, box2):
	"""We assume that the box follows the format: 
	box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4]
	where (x1,y1) and (x3,y3) represent the top left coordinate, 
	and (x2,y2) and (x4,y4) represent the bottom right coordinate.
	"""

	x1, y1, x2, y2 = box1
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou


# https://github.com/vineeth2309/Non-Max-Suppression/blob/main/NMS.py
def NMS(boxes, conf_threshold=0.7, iou_threshold=0.4):
	""" The function performs nms on the list of boxes:
		boxes: [box1, box2, box3...]
		box1: [x1, y1, x2, y2, Class, Confidence]."""

	bbox_list_thresholded = []	# List to contain the boxes after filtering by confidence
	bbox_list_new = []			# List to contain final boxes after nms 
	
	# Stage 1: (Sort boxes, and filter out boxes with low confidence)
	boxes_sorted = sorted(boxes, reverse=True, key = lambda x : x[5])	# Sort boxes according to confidence
	for box in boxes_sorted:
		if box[5] > conf_threshold:		# Check if the box has a confidence greater than the threshold
			bbox_list_thresholded.append(box)	# Append the box to the list of thresholded boxes 
		else:
			pass
	
	#Stage 2: (Loop over all boxes, and remove boxes with high IOU)
	while len(bbox_list_thresholded) > 0:
		current_box = bbox_list_thresholded.pop(0)		# Remove the box with highest confidence
		bbox_list_new.append(current_box)				# Append it to the list of final boxes
		for box in bbox_list_thresholded:
			if current_box[4] == box[4]:				# Check if both boxes belong to the same class
				iou = IOU(current_box[:4], box[:4])		# Calculate the IOU of the two boxes
				if iou > iou_threshold:					# Check if the iou is greater than the threshold defined
					bbox_list_thresholded.remove(box)	# If there is significant overlap, then remove the box
	
	return bbox_list_new



