import torch


def load_yolov5(weights_path):
    model = torch.hub.load(
        repo_or_dir = "external/yolov5/", 
        model = "custom", 
        source = "local", 
        path = weights_path, 
        force_reload = True)
    model.eval
    return model


__inference_models__ = {
    "YOLOv5": load_yolov5,
}

