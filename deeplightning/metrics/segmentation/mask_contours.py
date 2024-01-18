from typing import List, Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms import functional as F
import seaborn as sn
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2
import wandb

from deeplightning import METRIC_REGISTRY


__all__ = [
    "SemanticSegmentationMaskContour",
    "segmentation_mask_contour",
]


class SemanticSegmentationMaskContour():
    """Classification Accuracy metric class, inheriting from torchmetrics.

    Attributes (mandatory):
        display_name: name used by the logger when displaying the metric.
        logging_methods: metric methods called by the logger.
    """
    def __init__(self, cfg: OmegaConf):
        self.display_name = "segmentation_contour"
        self.logging_methods = ["draw"]

        self.grid_size = 4
        self.nrow = 2
        self.image_size = (200, 200)
        self.colors = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255], }


    def make_overlay(
        self,
        image_fp: str,
        mask_true_fp: str,
        mask_pred: Tensor,
        resize: Tuple[int,int] = None,
    ):
        """
        """
        if resize is not None:
            h_target, w_target = resize[0], resize[1]
            
        # Image
        image = cv2.imread(image_fp)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize is None:
            h_target, w_target, _ = image.shape
        image = cv2.resize(image, (w_target, h_target))
        
        # True Mask
        mask_true = cv2.imread(mask_true_fp)
        mask_true = cv2.resize(mask_true, (w_target, h_target))
        edges_true = cv2.Canny(mask_true, 50, 200)  # edge detector
        
        # Predicted Mask
        mask_pred = torch.argmax(mask_pred, dim=0).numpy()
        mask_pred = cv2.convertScaleAbs(mask_pred)
        mask_pred = cv2.resize(mask_pred, (w_target, h_target))
        mask_pred = 255 * mask_pred
        edges_pred = cv2.Canny(mask_pred, 50, 200)  # edge detector
        
        # Overlay
        overlay = image.copy()
        overlay[edges_true == 255] = self.colors["green"]
        overlay[edges_pred == 255] = self.colors["blue"]

        #plt.imshow(edges_true); plt.show()
        #plt.imshow(edges_pred); plt.show()
        #plt.imshow(overlay); plt.show()
        
        return torch.tensor(overlay).permute(2,0,1)


    def make_plot(
        self,
        image_fps: List[str],
        true_mask_fps: List[str],
        pred_masks: Tensor,
        resize: Tuple[int,int] = None,
    ):
        """
        """
        batch_indices = np.random.randint(len(image_fps), size=self.grid_size)
        batch_tensor = torch.stack([
            self.make_overlay(
                image_fp = image_fps[i], 
                mask_true_fp = true_mask_fps[i], 
                mask_pred = pred_masks[i,:,:,:],
                resize = resize,
            ) for i in batch_indices
        ])
        grid_image = make_grid(batch_tensor, nrow=self.nrow, padding=1)
        
        fig = plt.figure(figsize=(5,5))
        plt.imshow(grid_image.permute(1,2,0))
        plt.axis('off')
        plt.show()

        return fig
    

    def draw(self, 
        image_fps: str,
        true_mask_fps: str,
        preds: Tensor,  # segmentation masks
        resize: Tuple[int,int] = None,
    ):
        """Draw predicted and true mask contours overlaid on image.

        Args:
            image_fps: filep aths to images in the batch.
            true_mask_fps: file paths to masks in the batch.
            preds: mask predictions.
            resize: size of images when plotting.
        """
        # Draw figure
        figure = self.make_plot(
            image_fps = image_fps,
            true_mask_fps = true_mask_fps,
            pred_masks = preds,
            image_size = resize)
        
        # Save figure
        caption = f"Confusion Matrix [val, epoch {epoch+1}/{max_epochs}]"
        metric_tracker[logging_key] = wandb.Image(figure, caption=caption)


@METRIC_REGISTRY.register_element()
def segmentation_mask_contour(cfg: OmegaConf) -> SemanticSegmentationMaskContour:
    return SemanticSegmentationMaskContour(cfg)