""" Usage:
python  -m  scripts.pedestrian_detection.main  --conf_threshold 0.1  --model_flavour YOLO-v5  --model_cfg external/yolov5/models/yolov5n.yaml  --model_ckpt /Users/pme/data/_checkpoints/yolov5/yolov5n6.pt  --classes "person"  --input_path tools/pedestrian_detection/media/pexels-kate-trifo-4019405.jpg  --output_path /Users/pme/Downloads/tests/
python  -m  scripts.pedestrian_detection.main  --conf_threshold 0.1  --model_flavour YOLO-v5  --model_cfg external/yolov5/models/yolov5x.yaml  --model_ckpt /Users/pme/data/_checkpoints/yolov5/yolov5x6.pt  --classes "person"  --input_path tools/pedestrian_detection/media/pexels-kate-trifo-4019405.jpg  --output_path /Users/pme/Downloads/tests/

python  -m  scripts.pedestrian_detection.main  --conf_threshold 0.1  --model_flavour YOLO-v5  --model_cfg external/yolov5/models/yolov5n.yaml  --model_ckpt /Users/pme/data/_checkpoints/yolov5/yolov5n6.pt  --classes "person"  --input_path /Users/pme/Downloads/pexels-people-walking-2670-cut.mp4  --output_path /Users/pme/Downloads/tests/

"""

import argparse

from inference.vision.object_detection import ObjectDetector, OBJECTS_DICT


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_flavour", type=str, help="name of the model")
    parser.add_argument("--model_cfg", type=str, help="path to model configuration yaml")
    parser.add_argument("--model_ckpt", type=str, help="path to model checkpoint/weights")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="confidence threshold for detections")
    parser.add_argument("--iou_threshold", type=float, default=0.45, help="iou threshold for detections")
    parser.add_argument("--classes", type=str, default="person", help="classes to detect (comma-separated) e.g. 'person,car'")
    parser.add_argument("--input_path", type=str, help="path to image/video")
    parser.add_argument("--output_path", type=str, help="path to save outputs")
    args = parser.parse_args()
    return args


class PedestrianDetector(ObjectDetector):
    """Pedestrian Detector class.
    
    This class inherits from `ObjectDetector` class.

    Parameters
    ----------
    model_flavour : model name
    model_cfg : path to model configuration
    model_ckpt : path to model checkpoint

    """
    def __init__(self, model_flavour, model_cfg, model_ckpt):
        super().__init__(model_flavour, model_cfg, model_ckpt)


    def run_inference(self, input_path, conf_threshold, iou_threshold, classes):
        self.infer(input_path, conf_threshold, iou_threshold, classes)


    def run_vizualizer(self, 
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
        self.vizualize(
            input_path = input_path, 
            output_path = output_path,
            show_label = show_label,
            show_counter = show_counter,
            blur_people = blur_people,
            color_by = color_by,
            label_fontsize_factor = label_fontsize_factor,
            label_linewidth_factor = label_linewidth_factor,
            label_width_factor = label_width_factor,
            label_heigth_factor = label_heigth_factor,
            output_format = output_format,
        )


if __name__ == "__main__":

    args = parse_command_line_arguments()

    detector = PedestrianDetector(
        model_flavour = args.model_flavour,
        model_cfg = args.model_cfg, 
        model_ckpt = args.model_ckpt,
        )

    detector.run_inference(
        input_path = args.input_path,
        conf_threshold = args.conf_threshold,
        iou_threshold = args.iou_threshold,
        #['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        classes = [OBJECTS_DICT[x] for x in args.classes.strip().replace(" ", "").split(',')],
        )

    detector.run_vizualizer(
        input_path = args.input_path, 
        output_path = args.output_path,
        show_label = True,
        show_counter = False,
        blur_people = False,
        color_by = "class",   #{object,class}
        label_fontsize_factor = 0.018,
        label_linewidth_factor = 0.003,
        label_width_factor = 0.8,
        label_heigth_factor = 1.4,
        output_format = "jpeg",
        )
