from typing import Tuple
import sys
sys.path.insert(0, "..")
from omegaconf import OmegaConf
import argparse
import os
import io
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from flask import Flask, jsonify, request
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from lightning import LightningModule

from deeplightning.init.imports import get_reference 


parser = argparse.ArgumentParser()
parser.add_argument("--artifact_path", type=str, help="artifact storage path, containing checkpoint ('last.ckpt') and train config ('cfg.yaml').")
parser.add_argument("--host", type=str, default="localhost", help="host address")
parser.add_argument("--port", type=int, default=5000, help="host port")
args = parser.parse_args()


def load_model(confcfgig: OmegaConf, ckpt_path: str) -> LightningModule:
    """
    Import LightningModule class and initialize model. 
    Note that `load_from_checkpoint()` method should 
    be used on class reference, not class instance.
    """
    lightning_module = get_reference(cfg.model.module)
    model = lightning_module.load_from_checkpoint(
        checkpoint_path = ckpt_path, 
        cfg = cfg, # input to LightningModule
    )
    return model


def image_processing(image_bytes) -> Tensor:
    my_transforms = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    image_transformed = my_transforms(image).unsqueeze(0) #TODO batch support needed here
    return image_transformed


def image_prediction(image: Tensor) -> Tuple[str]:
    outputs = model(image)
    pred_class = outputs.argmax(1)
    pred_prob = F.softmax(outputs, dim=1) 
    # TODO: batch support needed here
    pred_class_str = str(pred_class.item())
    pred_prob_str = str(pred_prob[0, pred_class.item()].item())
    return pred_class_str, pred_prob_str



# create app
app = Flask(__name__)

# The `artifact_path` will contain the config 
# file (.yaml) and the model checkpoint (.ckpt)
CONFIG_PATH = os.path.join(args.artifact_path, "cfg.yaml")
CHECKPOINT_PATH = os.path.join(args.artifact_path, "last.ckpt")

# load train config - required to load 
# model with correct parameters
cfg = OmegaConf.load(CONFIG_PATH)

# load pretrained model from checkpoint
model = load_model(cfg, CHECKPOINT_PATH)
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Note that the image input is passed with name "image" 
    via `curl` or `requests.post()`.
    """
    if request.method == 'POST':

        if request.files.get('image'):

            image_bytes = request.files['image'].read()
            image = image_processing(image_bytes)
            class_id, class_probs = image_prediction(image)

            return jsonify({'predicted_class': class_id,
                            'predicted_probability': class_probs,})

        else:
            return jsonify({"ERROR": "File with name 'image' not found."})


if __name__ == '__main__':
    """
    Externally Visible Server. If you run the server you will notice that 
    the server is only accessible from your own computer, not from any other 
    in the network. This is the default because in debugging mode a user of 
    the application can execute arbitrary Python code on your computer.
    If you have the debugger disabled or trust the users on your network, 
    you can make the server publicly available simply by adding host 0.0.0.0:
    ```
        flask run --host=0.0.0.0 
    ```
    Alternatively, set the host in the app:
    ```
        app.run(host="0.0.0.0")
    ```
    This tells your operating system to listen on all public IPs.
    ----------------------------------------------------------------
    source: Flask docs
    """
    app.debug = False
    app.run(host=args.host, port=args.port)


