import pytorch_lightning as pl
import shutil
import os
import sys
sys.path.append("..")
import pytest

from tests.helpers import DummyModel


TESTNAME = os.path.basename(__file__).split(".")[0]


def run_test():

    model = DummyModel()

    trainer = pl.Trainer(
        max_epochs = 1,
        logger = False,
        limit_train_batches = 2,
        limit_val_batches = 2,
        enable_model_summary = False,
        enable_progress_bar = False,
    )

    trainer.fit(model = model)
    trainer.validate(model = model, verbose = False)
    trainer.test(model = model, verbose = False)
    
    shutil.rmtree('checkpoints')


if __name__ == "__main__":

    try:
        run_test()
        pass_message(TESTNAME, "passed 1/1")
    except Exception as e:
        print(e)
        fail_message(TESTNAME, "passed 0/1")
    
