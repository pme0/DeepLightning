""" 
Usage
-----
python  tools/pedestrian_detection/pedestrian_detection.py  --model_cfg external/yolov5/models/yolov5x.yaml  --model_ckpt /Users/pme/code/yolov5/yolov5x6.pt  --classes "person"  --input_path tools/pedestrian_detection/media/pexels-luis-dalvan-1770808-zoom.jpg  --output_path /Users/pme/Downloads/tests/

"""

from typing import Union, Tuple
import os
import sys
# ugly hack to be able to load modules as `import external.yolov5.ABC`.
# add path two levels upstream, to the main project folder.
current = os.path.dirname(os.path.realpath(__file__))
parent1 = os.path.dirname(current)
parent2 = os.path.dirname(parent1)
sys.path.append(parent2)
print(f"Added to system path: '{parent2}'")

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
from deeplightning.utils.io import read_video
from deeplightning.utils.messages import info_message, warning_message, error_message
from external.yolov5.utils.general import non_max_suppression


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", type=str, default="external/yolov5/models/yolov5x.yaml", help="path to model configuration yaml")
    parser.add_argument("--model_ckpt", type=str, help="path to model checkpoint/weights; will be downloaded from torch hub if not found")
    parser.add_argument("--classes", type=str, help="comma-separated classes to detect")
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Performing inference on '{self.device}' device")

        self.model_cfg = model_cfg
        self.model_ckpt = model_ckpt
        self.model = torch.hub.load(
            repo_or_dir = "external/yolov5/", 
            model = "custom", 
            source = "local", 
            path = model_ckpt, 
            force_reload = True).eval().to(self.device)
        

    def infer(self, input_path, classes):

        self.classes = classes
        self.input_filename = input_path.split("/")[-1].split(".")[0]
        
        if self.input_type == "image":

            transforms = T.Compose([T.Resize((640,640)), T.ToTensor()])
            image_tensors = Image.open(input_path)
            image_tensors = transforms(image_tensors).unsqueeze(0)
            w, h = imagesize.get(input_path)

        elif self.input_type == "video":

            transforms = T.Compose([T.Resize((640,640))])
            image_tensors, _, _ = read_video(video_path = args.input_path)
            image_tensors = torch.from_numpy(image_tensors).permute(0,3,1,2) # (B,C,H,W)
            image_tensors = transforms(image_tensors)
            _, _, h, w = image_tensors.shape
            
        else:

            raise NotImplementedError


        print(image_tensors.shape)
        num_frames = image_tensors.shape[0]
        image_tensors = image_tensors.to(self.device)

        conf_thres=0.20     # confidence threshold
        iou_thres=0.45      # IoU threshold
        max_det = 1000      # maximum number of detections
        agnostic_nms = False

        pred = self.model(image_tensors)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#[0]

        # rescale bboxes to original image size
        for i in range(num_frames):
            pred[i][:,0] = pred[i][:,0] * w / 640
            pred[i][:,1] = pred[i][:,1] * h / 640
            pred[i][:,2] = pred[i][:,2] * w / 640
            pred[i][:,3] = pred[i][:,3] * h / 640

        # create bounding boxes list
        self.bboxes = [[] for _ in range(num_frames)]
        for frame in range(num_frames):
            for obj in pred[frame]:
                self.bboxes[frame].append({
                    "class": int(obj[5]),
                    "box": x0y0x1y1_to_x0y0wh(box=[float(x) for x in obj[0:4]]),
                    "conf": float(obj[4]),
                })


    def vizualize(self, 
        input_path: str, 
        output_path: str, 
        show_label: bool, 
        show_counter: bool, 
        blur_people: bool, 
        color_by: str,
        label_linewidth_factor: float,
        label_fontsize_factor: float,
        label_width_factor: float,
        label_heigth_factor: float,
    ):
        """Visualize
        """

        if self.input_type == "image":

            self.make_image_with_bboxes(
                image_path = input_path, 
                output_path = output_path,
                show_label = show_label,
                show_counter = show_counter,
                blur_people = blur_people, 
                color_by = color_by,
                label_linewidth_factor = label_linewidth_factor,
                label_fontsize_factor = label_fontsize_factor,
                label_width_factor = label_width_factor,
                label_heigth_factor = label_heigth_factor,
                )

        elif self.input_type == "video":

            self.make_video_with_bboxes()

        else:

            raise NotImplementedError


    def make_image_with_bboxes(self,
        image_path: str, 
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

        for j, frame in enumerate(self.bboxes):
            for i, bbox in enumerate(frame):

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
                plt.savefig(os.path.join(output_path, f"{self.input_filename}_bboxes_{j}.png"), bbox_inches='tight', pad_inches=0)
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
        classes = [OBJECTS_DICT[x] for x in args.classes.strip().replace(" ", "").split(',')]
        #['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        )

    detector.vizualize(
        input_path = args.input_path, 
        output_path = args.output_path,
        show_label = True,
        show_counter = False,
        blur_people = False,
        color_by = "object",
        label_fontsize_factor = 0.018,
        label_linewidth_factor = 0.004,
        label_width_factor = 0.8,
        label_heigth_factor = 1.4,
        )
