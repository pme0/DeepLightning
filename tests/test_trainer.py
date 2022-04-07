import os
import sys
sys.path.insert(0, "..")
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import pytest

from tests.helpers.tools import compare_model_params
from tests.helpers.dummies import DummyModel
from torch.utils.data import DataLoader, Dataset
from tests.helpers.housekeeping import clear_temporary_dir
from deeplightning.config.load import load_config
from deeplightning.init.initializers import init_model, init_trainer

FILE_NAME = os.path.basename(__file__).split(".")[0]
TMP_DIR = "tmp"
CKPT_PATH = os.path.join(TMP_DIR, "last.ckpt")


def test_initialization():

    config = load_config(config_file = "helpers/dummy_config.yaml")
    model = init_model(config)
    trainer = init_trainer(config)

    assert OmegaConf.is_config(config)
    assert isinstance(model, pl.LightningModule)
    assert isinstance(trainer, pl.Trainer)


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
    model = init_model(config)
    trainer = pl.Trainer(
        max_epochs = 1,
        logger = False,
        strategy = kwargs['strategy'], 
        gpus = kwargs['gpus'],
        precision = kwargs['precision'],
        limit_train_batches = 2,
        limit_val_batches = 2,
        enable_model_summary = False,
        enable_progress_bar = False,
        enable_checkpointing = False,
        )

    trainer.fit(model)

    assert trainer.state.finished, f"Trainer state: {trainer.state}"
    assert trainer.current_epoch == (trainer.max_epochs-1)