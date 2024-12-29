import argparse
import io

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
import uvicorn

from api.helpers import get_config, get_checkpoint, get_model
from deeplightning.datasets.info import DATASET_INFO


app = FastAPI()


parser = argparse.ArgumentParser()
parser.add_argument(
    "run_dir",
    type=str,
    help="artifact storage directory",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default=None,
    help="checkpoint name",
)
parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="host address",
)
parser.add_argument(
    "--port",
    type=int,
    default=5000,
    help="host port",
)
parser.add_argument(
    "--reload",
    action="store_true",
    help="Enable auto-reload during development",
)
args = parser.parse_args()


cfg = get_config(args.run_dir)
ckpt = get_checkpoint(args.run_dir, args.ckpt_name)
model = get_model(ckpt, cfg)

apply_greyscale = cfg.task.model.args["num_channels"] == 1
resize = cfg.data.transforms.test.get(
    "resize", 
    DATASET_INFO[cfg.data.dataset].image_size
)

transforms = T.Compose([
    T.Grayscale() if apply_greyscale else None,
    T.Resize(resize),
    T.ToTensor(),
    T.Normalize(
        mean=cfg.data.transforms.test["normalize"]["mean"], 
        std=cfg.data.transforms.test["normalize"]["std"]),
])


def preprocess(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transforms(image)
    image = image.unsqueeze(0)
    return image


def predict(image_tensor: torch.Tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()


@app.post("/predict/")
async def predict_image(file: UploadFile):
    
    try:
        image_bytes = await file.read()
        image_tensor = preprocess(image_bytes)
        predicted_class = predict(image_tensor)
        
        return JSONResponse(
            content={
                "predicted_class": predicted_class,
                "image": file.filename,
        })
    
    except Exception as e:

        return JSONResponse(
            status_code=400, 
            content={"error": str(e),
        })


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
    