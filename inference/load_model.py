import torch


def load_inference_model(model_flavour: str, params: dict, device: torch.device):
    """General function for loading inference model.

    Parameters
    ----------
    model_flavour : the type of model to be loaded. It must match one 
        of the keys in `INFERENCE_MODELS`
    params : the input parameters to the model loader
    device : the computing device to be used for inference

    """
    return INFERENCE_MODELS[model_flavour](**params).eval().to(device)


def load_YOLOv5(model_ckpt):
    model = torch.hub.load(
        repo_or_dir = "external/yolov5/", 
        model = "custom", 
        source = "local", 
        path = model_ckpt, 
        force_reload = True)
    return model


INFERENCE_MODELS = {
    "YOLO-v5": load_YOLOv5,
}
