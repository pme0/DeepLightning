import os
import sys
sys.path.insert(0, "..")
import torch
import pytorch_lightning as pl
import pytest

from tests.helpers.runif import single_gpu_available, multi_gpu_available
from tests.helpers.tools import DummyModel
from tests.helpers.housekeeping import clear_temporary_dir


FILE_NAME = os.path.basename(__file__).split(".")[0]
TMP_DIR = "tmp"
CKPT_PATH = os.path.join(TMP_DIR, "last.ckpt")


def setup_trainer(strategy, precision, gpus):
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath = TMP_DIR, 
            save_last = True, 
            every_n_epochs = 1)

    trainer = pl.Trainer(
        max_epochs = 1,
        logger = False,
        strategy = strategy, 
        precision = precision, 
        gpus = gpus,
        limit_train_batches = 2,
        limit_val_batches = 2,
        enable_model_summary = False,
        enable_progress_bar = False,
        callbacks = [checkpoint_callback],
        )

    return trainer


def compare_model_params(modelA, modelB):
    passed = True
    for key in modelA.state_dict():
        if (modelA.state_dict()[key] != modelB.state_dict()[key]).sum().item() > 0:
            passed = False
    return passed


@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param(
            values = dict(strategy = None,  precision = 32, gpus = None)),

        pytest.param(
            values = dict(strategy = "ddp", precision = 32, gpus = [0]), 
            marks = pytest.mark.skipif(
                condition = not torch.cuda.is_available(), 
                reason="single-gpu unavailable")),

        pytest.param(
            values = dict(strategy = "ddp", precision = 32, gpus = [0,1]), 
            marks = pytest.mark.skipif(
                condition = torch.cuda.device_count() < 2, 
                reason="multi-gpu unavailable")
        ),
    )
)
def test_checkpointing(kwargs):

    clear_temporary_dir(TMP_DIR)

    model = DummyModel()
    trainer = setup_trainer(**kwargs)
    trainer.fit(model)
    model_ = DummyModel.load_from_checkpoint(CKPT_PATH)
    passed = compare_model_params(model, model_)

    clear_temporary_dir(TMP_DIR)

    assert trainer.state.finished, f"Trainer state: {trainer.state}."
    assert passed, "Model parameters differ between saved and loaded model."
    