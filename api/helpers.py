import glob
import os

from omegaconf import OmegaConf

from deeplightning.core.dlconfig import DeepLightningConfig, reload_config
from deeplightning.tasks.vision.classification import ImageClassificationTask
from deeplightning import TASK_REGISTRY


def get_config(run_dir: str) -> DeepLightningConfig:
    cfg_fp = os.path.join(run_dir, "files/cfg.yaml")
    cfg = OmegaConf.load(cfg_fp)
    cfg = reload_config(cfg)
    return cfg


def get_checkpoint(run_dir: str, ckpt_name: str):
    if ckpt_name:
        return os.path.join(run_dir, "files", ckpt_name)
    else:
        ckpt_files = glob.glob(os.path.join(run_dir, "files/*.ckpt"))
        if len(ckpt_files) > 1:
            raise RuntimeError(
                f"Multiple checkpoints found in directory '{run_dir}': "
                f"'{ckpt_files}'. Specify which checkpoint to use using "
                f"argument 'ckpt_name'."
            )
        return ckpt_files[0]


def get_model(ckpt: str, cfg: DeepLightningConfig):
    task = cfg.task.name

    if task == "image_classification":
        model = ImageClassificationTask.load_from_checkpoint(ckpt, cfg=cfg)
    else:
        raise NotImplementedError(
            f"Please implement an API for task '{task}'."
        )
    
    model.eval()

    return model