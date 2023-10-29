import os
import sys
sys.path.insert(0, "..")
from omegaconf import OmegaConf
import torch
import lightning as pl
import pytest

from deeplightning.config.load import load_config
from deeplightning.init.initializers import init_model, init_trainer
from deeplightning.trainer.trainer import DLTrainer


FILE_NAME = os.path.basename(__file__).split(".")[0]
TMP_DIR = "tmp"
CKPT_PATH = os.path.join(TMP_DIR, "last.ckpt")


@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param(
            dict(accelerator="cpu", strategy="auto", devices="auto", precision=32)),
        pytest.param(
            dict(accelerator="gpu", strategy="ddp", devices="auto", precision=32), 
            marks = pytest.mark.skipif(
                condition = not torch.cuda.is_available(), 
                reason="gpu unavailable")),
        pytest.param(
            dict(accelerator="gpu", strategy="ddp", precision=32), 
            marks = pytest.mark.skipif(
                condition = torch.cuda.device_count() < 2, 
                reason="multi-gpu unavailable")),
    )
)
def test_trainer(kwargs):

    cfg = load_config(config_file = "tests/helpers/_dummy.yaml")
    
    cfg.engine.accelerator = kwargs["accelerator"]
    cfg.engine.strategy = kwargs["strategy"]
    cfg.engine.precision = kwargs["precision"]
    cfg.engine.devices = kwargs["devices"]
    
    model = init_model(cfg)
    trainer = init_trainer(cfg)

    assert OmegaConf.is_config(cfg)
    assert isinstance(model, pl.LightningModule)
    assert isinstance(trainer, pl.Trainer)
    assert isinstance(trainer, DLTrainer)

    trainer.fit(model)

    assert trainer.state.finished, f"Trainer state: {trainer.state}"
    assert trainer.current_epoch == (trainer.max_epochs-1)

