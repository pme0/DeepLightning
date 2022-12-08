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


# https://github.com/vineeth2309/Non-Max-Suppression/blob/main/NMS.py

def IOU(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate."""

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



