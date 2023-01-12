""" 
Usage:
    python  tools/pedestrian_detection/main.py  --model_cfg external/yolov5/models/yolov5x.yaml  --model_ckpt /Users/pme/code/yolov5/yolov5x6.pt  --input_path /Users/pme/Downloads/pexels-kate-trifo-4019405.jpg  --output_path /Users/pme/Downloads/tests/

Sources:
    - https://www.pexels.com/photo/people-walking-on-pedestrian-lane-4019405/
    - 

"""

import os
import sys
# ugly hack to be able to load modules as `import external.yolov5.ABC`.
# add path two levels upstream, to the main project folder.
current = os.path.dirname(os.path.realpath(__file__))
parent1 = os.path.dirname(current)
sys.path.append(parent1)
print(f"Added to system path: '{parent1}'")

from typing import List, Dict
import numpy as np
import time
import math
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter
import imagesize
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse

from deeplightning.utils.detection.bbox_converter import x0y0x1y1_to_x0y0wh
from deeplightning.utils.messages import info_message, warning_message, error_message
from external.yolov5.utils.general import non_max_suppression


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", type=str, default="external/yolov5/models/yolov5x.yaml", help="path to model configuration yaml")
    parser.add_argument("--model_ckpt", type=str, help="path to model checkpoint/weights; will be downloaded from torch hub if not found")
    parser.add_argument("--input_path", type=str, help="path to image/video")
    parser.add_argument("--output_path", type=str, help="path to save outputs")
    args = parser.parse_args()
    return args


class PedestrianDetector():
    """
    """

    def __init__(self, model_cfg, model_ckpt, input_path):

        if isinstance(input_path, str):
            if input_path.endswith(IMG_EXT):
                self.input_type = "image"
            elif input_path.endswith(VID_EXT):
                self.input_type = "video"
            else:
                raise NotImplementedError
        else:
            raise ValueError

        self.model_cfg = model_cfg
        self.model_ckpt = model_ckpt
        self.model = torch.hub.load(
            repo_or_dir = "external/yolov5/", 
            model = "custom", 
            source = "local", 
            path = model_ckpt, 
            force_reload = True).eval()
        

    def infer(self, input_path, classes):

        self.classes = classes
        
        if self.input_type == "image":

            w, h = imagesize.get(input_path)
            image = Image.open(input_path)
            transforms = T.Compose([T.Resize((640,640)), T.ToTensor()])
            image_tensors = transforms(image).unsqueeze(0)

        elif self.input_type == "video":

            raise NotImplementedError
            
        else:

            raise NotImplementedError


        conf_thres=0.20     # confidence threshold
        iou_thres=0.45      # IoU threshold
        max_det = 1000      # maximum number of detections
        agnostic_nms = False

        pred = self.model(image_tensors)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        # rescale bboxes to original image size
        pred[:,0] = pred[:,0] * w / 640
        pred[:,1] = pred[:,1] * h / 640
        pred[:,2] = pred[:,2] * w / 640
        pred[:,3] = pred[:,3] * h / 640

        self.bboxes = []
        for obj in pred:
            self.bboxes.append(
            {
                "class": int(obj[5]), 
                "box": x0y0x1y1_to_x0y0wh(box=[float(x) for x in obj[0:4]]),
                "conf": float(obj[4]), 
            })


    def vizualize(
        self, 
        input_path, 
        output_path, 
        resize, 
        show_label, 
        show_counter, 
        blur_people, 
        color_by,
        label_linewidth_factor,
        label_fontsize_factor,
        label_width_factor,
        label_heigth_factor,
    ):
        """
        """
        if self.input_type == "image":

            self.plot_image_and_bboxes(
                image_path = input_path, 
                output_path = output_path,
                resize = resize, 
                show_label = show_label,
                show_counter = show_counter,
                blur_people = blur_people, 
                color_by = color_by,
                label_linewidth_factor = label_linewidth_factor,
                label_fontsize_factor = label_fontsize_factor,
                label_width_factor = label_width_factor,
                label_heigth_factor = label_heigth_factor,
                )
        else:

            raise NotImplementedError


    def plot_image_and_bboxes(
        self,
        image_path: str, 
        resize: int = None, 
        output_path: str = None, 
        show_label: bool = True,
        show_counter: bool = False,
        blur_people: bool = False,
        color_by: str = "object",
        label_linewidth_factor: float = 0.002,
        label_fontsize_factor: float = 0.015,
        label_width_factor: float = 0.08,
        label_heigth_factor: float = 1.4,
        ):
        """Plot an image together with a set of bounding boxes and class labels

        Example:
        ```
        from deeplightning.viz.image.bboxes import plot_image_and_bboxes
        img_path = "media/eye.jpg"
        bboxes = [
            {"class": "iris",  "box": [253, 245, 244, 240], "format": "xcycwh"},
            {"class": "pupil", "box": [244, 243, 68+x, 64], "format": "xcycwh"}]
        plot_image_and_bboxes(image_path=img_path, bboxes=bboxes, resize=500, 
                        output_path=None, show_image=True)
        ```
        """
        assert color_by in ("object", "class")
       
        image = Image.open(image_path)
        img_w, img_h = imagesize.get(image_path)

        # params
        colors = sn.color_palette("Set2")
        color_class = {self.classes[i]: colors[i] for i in range(len(self.classes))}
        label_linewidth = label_linewidth_factor * img_w
        label_fontsize = label_fontsize_factor * img_w
        label_width_size = label_width_factor * label_fontsize
        label_heigth_size = label_heigth_factor * label_fontsize
        
        # plot
        # figure size is set in pixels; by default some pixels in each 
        # direction are allocated to white border/axes, so adjustment for that 
        # is required if the saved image is to be exactly `resize x resize`;
        resize_extra = 1.3 * img_w
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html#figure-size-in-pixel
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1,1, figsize=(resize_extra*px, resize_extra*px))  
        
        # blur bboxes
        if blur_people:
            for i, bbox in enumerate(self.bboxes):
                cbox = (math.floor(bbox["box"][0])-1, math.floor(bbox["box"][1])-1, math.ceil(bbox["box"][0]+bbox["box"][2])+1, math.floor(bbox["box"][1]+bbox["box"][3])+1) #left, upper, right, and lower
                ic = image.crop(cbox)
                for i in range(3):  # use BLUR filter multiple times to get the desired effect
                    ic = ic.filter(ImageFilter.BLUR)
                image.paste(ic, cbox)

        plt.imshow(image)
            
        for i, bbox in enumerate(self.bboxes):

            # boundings box
            ax.add_patch(
                patches.Rectangle(
                    xy=(bbox["box"][0], bbox["box"][1]), 
                    width=bbox["box"][2],
                    height=bbox["box"][3], 
                    edgecolor=colors[i%len(colors)] if color_by == "object" else color_class[bbox["class"]],
                    facecolor='none', 
                    alpha=1, 
                    linewidth=label_linewidth,
                )
            )
            
            # label box
            if show_label:
                label_conf = None if "conf" not in bbox else bbox["conf"]
                label_text = "{}{}{:.2f}".format(OBJECTS[bbox["class"]].lower(), "" if label_conf is None else " ", round(label_conf,2))
                ax.add_patch(
                    patches.Rectangle(
                        xy=(bbox["box"][0], bbox["box"][1]+bbox["box"][3] - label_heigth_size), 
                        width=label_width_size * len(label_text), 
                        height=label_heigth_size, 
                        facecolor=colors[i%len(colors)] if color_by == "object" else color_class[bbox["class"]], 
                        edgecolor='none', 
                        alpha=0.6,
                    )
                )

            # label text
            if show_label:
                plt.annotate(
                    text=label_text, 
                    xy=(bbox["box"][0], bbox["box"][1]+bbox["box"][3]),
                    color='white', alpha=1, font=dict(size=label_fontsize),
                    ha="left", va = "bottom",
                    clip_on=True, # avoids annotation outside the plot area, which cretes a white margin that cannot be trimmed by off-ing the axes
                )

        # counter
        if show_counter:
            ax.add_patch(
                patches.Rectangle(
                    xy=(0, 0), width=0.15*img_h, height=0.15*img_h, 
                    facecolor=(16/255, 173/255, 237/255), edgecolor='none', alpha=0.6,
                )
            )
            plt.annotate(
                text="counter", 
                xy=(0.075*img_h, 0.02*img_h),
                color='white', alpha=1, font=dict(size=0.8*label_fontsize),
                ha="center", va = "top",
                clip_on=True, # avoids annotation outside the plot area, which cretes a white margin that cannot be trimmed by off-ing the axes
            )
            plt.annotate(
                text=len(self.bboxes), 
                xy=(0.075*img_h, 0.075*img_h),
                color='white', alpha=1, font=dict(size=1.1*label_fontsize),
                ha="center", va = "center",
                clip_on=True, # avoids annotation outside the plot area, which cretes a white margin that cannot be trimmed by off-ing the axes
            )

        # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image
        plt.gca().set_axis_off()

        if output_path is not None:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, "test.png"), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":

    IMG_EXT = (".png", ".jpg", ".jpeg")
    VID_EXT = (".mp4")
    OBJECTS = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    OBJECTS_DICT_INV = {i: OBJECTS[i] for i in range(len(OBJECTS))}
    OBJECTS_DICT = {v: k for k, v in OBJECTS_DICT_INV.items()}

    args = parse_command_line_arguments()

    detector = PedestrianDetector(
        model_cfg = args.model_cfg, 
        model_ckpt = args.model_ckpt,
        input_path = args.input_path)

    detector.infer(
        input_path = args.input_path,
        classes = [OBJECTS_DICT[x] for x in ['person']]
        #['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        )

    detector.vizualize(
        input_path = args.input_path, 
        output_path = args.output_path,
        resize = None,
        show_label = True,
        show_counter = False,
        blur_people = False,
        color_by = "object",
        label_fontsize_factor = 0.018,
        label_linewidth_factor = 0.004,
        label_width_factor = 0.8,
        label_heigth_factor = 1.4,
        )
