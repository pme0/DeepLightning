import argparse

import uvicorn

from api.helpers import get_config


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


def serve_app():

    cfg = get_config(args.run_dir)
    task = cfg.task.name

    if task == "image_classification":
        from api.apps import img_cls
        uvicorn.run(img_cls.app, host=args.host, port=args.port)
    else:
        raise NotImplementedError(
            f"Please implement an API for task '{task}'."
        )

if __name__ == "__main__":
    serve_app()
