import os
import sys
sys.path.insert(0, "..")
import torch
import lightning as pl
import pytest

from tests.helpers.tools import compare_model_params
from tests.helpers.dummies import DummyModel
from tests.helpers.housekeeping import clear_temporary_dir


FILE_NAME = os.path.basename(__file__).split(".")[0]
TMP_DIR = "tmp"
CKPT_PATH = os.path.join(TMP_DIR, "last.ckpt")


def setup_trainer(accelerator, strategy, devices, precision):

    ckpt_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath = TMP_DIR, 
            save_last = True, 
            every_n_epochs = 1)

    trainer = pl.Trainer(
        max_epochs = 1,
        logger = False,
        accelerator = accelerator,
        strategy = strategy, 
        devices = devices,
        precision = precision, 
        limit_train_batches = 2,
        limit_val_batches = 2,
        enable_model_summary = False,
        enable_progress_bar = False,
        callbacks = [ckpt_callback],
        )

    return trainer


@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param(
            dict(accelerator="cpu", strategy="auto", devices="auto", precision=16)),
        pytest.param(
            dict(accelerator="cpu", strategy="auto", devices="auto", precision=32)),
        pytest.param(
            dict(accelerator="gpu", strategy="auto", devices=[0], precision=16),
            marks = pytest.mark.skipif(
                condition = not torch.cuda.is_available(), 
                reason="gpu unavailable")),
        pytest.param(
            dict(accelerator="gpu", strategy="auto", devices=[0], precision=32),
            marks = pytest.mark.skipif(
                condition = not torch.cuda.is_available(), 
                reason="gpu unavailable")),
        pytest.param(
            dict(accelerator="gpu", strategy="auto", devices=[0,1], precision=16),
            marks = pytest.mark.skipif(
                condition = torch.cuda.device_count() < 2, 
                reason="multi-gpu unavailable")),
        pytest.param(
            dict(accelerator="gpu", strategy="auto", devices=[0,1], precision=32), 
            marks = pytest.mark.skipif(
                condition = torch.cuda.device_count() < 2, 
                reason="multi-gpu unavailable")),
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
    