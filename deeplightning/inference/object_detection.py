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
import time
import math
import numpy as np
#from models.yolo import Model
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter
import imagesize
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse

from external.yolov5.utils.general import non_max_suppression
from deeplightning.utils.detection.bbox_converter import x0y0x1y1_to_x0y0wh
from deeplightning.utils.messages import info_message, warning_message, error_message


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", type=str, default="external/yolov5/models/yolov5x.yaml", help="path to model configuration yaml")
    parser.add_argument("--model_weights", type=str, help="path to model checkpoint/weights; will be downloaded from torch hub if not found")
    parser.add_argument("--source", type=str, help="path to image/video")
    args = parser.parse_args()
    return args


def plot_image_and_bboxes(
    image_path: str, 
    bboxes: List[List[Dict]] = None,
    resize: int = None, 
    save_path: str = None, 
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
                    save_path=None, show_image=True)
    ```
    """
    sncolors = sn.color_palette("Set2") #sncolors[6]
    colors = {0: (255/255, 198/255, 0/255) , 1: sncolors[1]} #colors[0] #(255/255, 198/255, 0/255) 
    colors = sncolors
    
    image = Image.open(image_path)
    img_w, img_h = imagesize.get(image_path)
    assert img_w == img_h  # assumes square image
    if resize is None:
        resize = img_w
    else:
        image = image.resize((resize, resize))
        
    # params
    font_factor = 0.02
    label_width_factor = 0.00005 * img_h
    label_heigth_factor = 0.00011 * img_h
    
    # plot
    # figure size is set in pixels;
    # be default it seems 30 pixels in each direction are allocated 
    # to white border/axes, so adjustment for that is required if
    # the saved image is to be exactly `resize x resize`;
    px = 1/plt.rcParams['figure.dpi']
    resize_extra = 30 + resize
    fig, ax = plt.subplots(1,1, figsize=(resize_extra*px, resize_extra*px))    
    plt.imshow(image)
        
    for i, bbox in enumerate(bboxes):

        # boundings box

        ax.add_patch(
            patches.Rectangle(
                xy=(bbox["box"][0], bbox["box"][1]), 
                width=bbox["box"][2], 
                height=bbox["box"][3], 
                facecolor='none',
                edgecolor=colors[i%len(colors)], #colors[box["class"]],
                linewidth=1.5,
                alpha=1,
            )
        )
        
        # label box
        label_conf = None if "conf" not in bbox else bbox["conf"]
        label_text = "{}{}{:.2f}".format(OBJECTS[bbox["class"]].lower(), "" if label_conf is None else " ", round(label_conf,2))
        label_width = label_width_factor * resize_extra * len(label_text)
        if bbox["box"][0] + label_width > img_w: # avoid plotting outside image
            pass
        label_heigth = label_heigth_factor * resize_extra
        if bbox["box"][1] + label_heigth > img_h: # avoid plotting outside image
            pass
        ax.add_patch(
            patches.Rectangle(
                xy=(bbox["box"][0], bbox["box"][1]+bbox["box"][3] - label_heigth), 
                width=label_width, 
                height=label_heigth, 
                facecolor=colors[i%len(colors)], #colors[box["class"]]
                edgecolor='none',
                linewidth=2,
                alpha=0.6,
            )
        )
        # label text
        fontsize = font_factor * img_h
        plt.annotate(
            text=label_text, 
            xy=(bbox["box"][0], bbox["box"][1]+bbox["box"][3]),
            color='white',
            ha="left",
            va = "bottom",
            alpha=1,
            font=dict(size=fontsize),
        )
    # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image
    plt.gca().set_axis_off()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
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

    args = parse_command_line_arguments()

    model = torch.hub.load(
        repo_or_dir = "external/yolov5/", 
        model = "custom", 
        source = "local", 
        path = args.model_weights, 
        force_reload = True)
    model.eval

    if args.source.endswith(IMG_EXT):
        w, h = imagesize.get(args.source)
        image = Image.open(args.source)
        transforms = T.Compose([T.Resize(640), T.ToTensor()])
        image_tensors = transforms(image).unsqueeze(0)

    elif args.source.endswith(VID_EXT):
        pass

    else:
        raise NotImplementedError

 
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45
    classes = 0 # filter
    agnostic_nms = False
    max_det = 1000

    pred = model(image_tensors)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    pred = pred * w / 640

    bboxes = []
    for obj in pred:
        bboxes.append(
        {
            "class": int(obj[5]), 
            "box": x0y0x1y1_to_x0y0wh(box=[float(x) for x in obj[0:4]]),
            "conf": float(obj[4]), 
        })

    plot_image_and_bboxes(
        image_path = "/Users/pme/Downloads/people.jpeg", 
        bboxes = bboxes,
        resize = None, 
        save_path = "/Users/pme/Downloads/try.png", 
        )