import os
import math
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter
import imagesize
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import patches

from deeplightning.utils.io_local import read_video
from deeplightning.utils.detection.bbox_converter import x0y0x1y1_to_x0y0wh
from inference.load_model import load_inference_model
from external.yolov5.utils.general import non_max_suppression


#TODO move these variables into another file - use `OBJECTS = DETECTED_OBJECTS[model_flavour]`
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

#TODO move this class into another file
class DataSequence():
    """
    """
    def __init__(self, input_path):
        self.input_path = input_path
        self.transforms = T.Compose([T.Resize((640,640))])
        self.excludes = (".DS_Store",)

        self.get_input_type()
        self.get_tensors()

    
    def get_input_type(self):
        """
        """
        if isinstance(self.input_path, str):
            if self.input_path.endswith(IMG_EXT):
                self.input_type = "image"
            elif self.input_path.endswith(VID_EXT):
                self.input_type = "video"
            else:
                raise NotImplementedError
        else:
            raise ValueError


    def get_tensors(self):
        """
        """
        if self.input_type == "image":
            self.image_input_handler()
        elif self.input_type == "video":
            self.video_input_handler()


    def image_input_handler(self):
        """
        """
        self.image_tensors = Image.open(self.input_path).convert("RGB")
        self.w, self.h = self.image_tensors.size
        self.image_tensors =  T.ToTensor()(self.image_tensors).unsqueeze(0)
        print("Image tensor shape (original image):", tuple(self.image_tensors.shape))
        self.image_tensors = self.transforms(self.image_tensors)


    def video_input_handler(self):
        """
        """
        self.image_tensors, _, _ = read_video(video_path=self.input_path)
        self.w, self.h = self.image_tensors.shape[2], self.image_tensors.shape[1]
        self.image_tensors = torch.from_numpy(self.image_tensors).permute(0,3,1,2) # (B,C,H,W)
        print("Video tensor shape (original video):", tuple(self.image_tensors.shape))
        self.image_tensors = self.transforms(self.image_tensors)



class ObjectDetector():
    """Base Object Detector class.

    Parameters
    ----------
    model_flavour : model name
    model_cfg : path to model configuration
    model_ckpt : path to model checkpoint
    device: 
    """

    def __init__(self, model_flavour, model_cfg, model_ckpt, device=None):
        
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ) if device is None else torch.device(device)
        print(f"Inference device: {self.device}")

        self.model_cfg = model_cfg
        self.model_ckpt = model_ckpt
        self.model = load_inference_model(
            model_flavour = model_flavour, 
            params = {'model_ckpt': model_ckpt},
            device = device,
        )


    def infer(self, input_path, conf_threshold, iou_threshold, classes):

        self.data = DataSequence(input_path)
        w, h = self.data.w, self.data.h
        self.classes = classes
        self.input_filename = input_path.split("/")[-1].split(".")[0]

        print("Image tensor shape (after converting to model input requirements):", tuple(self.data.image_tensors.shape))
        num_frames = self.data.image_tensors.shape[0]
        self.data.image_tensors = self.data.image_tensors.to(self.device)

        conf_thres=conf_threshold   # Confidence threshold
        iou_thres=iou_threshold     # IoU threshold
        max_det = 1000              # maximum number of detections
        agnostic_nms = False

        pred = self.model(self.data.image_tensors)
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
        output_format: str,
    ):
        """Visualize
        """

        if self.data.input_type == "image":

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
                output_format = output_format,
                )

        elif self.data.input_type == "video":
            
            self.make_video_with_bboxes()  #save video instead of frames

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
        output_format: str = "png",
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
        
        min_conf_for_label_box_and_text = 0.2

        # plot
        # figure size is set in pixels; by default some pixels in each 
        # direction are allocated to white border/axes, so adjustment for that 
        # is required if the saved image is to be exactly `resize x resize`;
        #resize_extra = 1.3 * img_w
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html#figure-size-in-pixel
        #px = 1/plt.rcParams['figure.dpi']
        #fig, ax = plt.subplots(1,1, figsize=(resize_extra*px, resize_extra*px)) 
        
        # TO CREATE IMAGES THAT ARE OF FIXED SIZE
        # follow the approach in spectogramer.py in https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd 
        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches((img_w/dpi, img_h/dpi))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # blur bboxes
        if blur_people:
            for j, frame in enumerate(self.bboxes):
                for i, bbox in enumerate(frame):
                    cbox = (  #left, upper, right, and lower
                        math.floor(bbox["box"][0])-1, 
                        math.floor(bbox["box"][1])-1, 
                        math.ceil(bbox["box"][0]+bbox["box"][2])+1, 
                        math.floor(bbox["box"][1]+bbox["box"][3])+1
                    )
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
                    # avoid labelling unconfident predictiong, usually small objects
                    if label_conf > min_conf_for_label_box_and_text:
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
                    # avoid labelling unconfident predictiong, usually small objects
                    if label_conf > min_conf_for_label_box_and_text:
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
            #plt.gca().set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

            if output_path is not None:
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                plt.savefig(
                    fname = os.path.join(output_path, f"{self.input_filename}_bboxes_{j}.{output_format}"), 
                    bbox_inches = 'tight', 
                    pad_inches = 0,
                )
            plt.close()


    def make_video_with_bboxes():
        """
        """
        raise NotImplementedError("automate the (currently) manual process for video inputs")