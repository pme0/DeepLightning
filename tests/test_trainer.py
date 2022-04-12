import os
import sys
sys.path.insert(0, "..")
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
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
            dict(
                strategy = None, 
                precision = 32,
                gpus = None)),
        pytest.param(
            dict(
                strategy = "ddp",  
                precision = 32, 
                gpus = [0]), 
            marks = pytest.mark.skipif(
                condition = not torch.cuda.is_available(), 
                reason="single-gpu unavailable")),
        pytest.param(
            dict(
                strategy = "ddp",  
                precision = 32, 
                gpus = [0,1]), 
            marks = pytest.mark.skipif(
                condition = torch.cuda.device_count() < 2, 
                reason="multi-gpu unavailable")),
    )
)
def test_trainer(kwargs):

    config = load_config(config_file = "helpers/dummy_config.yaml")
    
    config.engine.backend = kwargs["strategy"]
    config.engine.precision = kwargs["precision"]
    config.engine.gpus = kwargs["gpus"]
    # TODO extra params for quick testing
    config.test_params.limit_train_batches = 2
    config.test_params.limit_val_batches = 2
    config.test_params.enable_model_summary = False,
    config.test_params.enable_progress_bar = False,
    config.test_params.logger = False
    
    model = init_model(config)
    trainer = init_trainer(config)

    assert OmegaConf.is_config(config)
    assert isinstance(model, pl.LightningModule)
    assert isinstance(trainer, pl.Trainer)
    assert isinstance(trainer, DLTrainer)

    trainer.fit(model)

    assert trainer.state.finished, f"Trainer state: {trainer.state}"
    assert trainer.current_epoch == (trainer.max_epochs-1)

