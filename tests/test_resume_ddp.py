import itertools
import os
import sys
sys.path.insert(0, "..")

import pytorch_lightning as pl
from helpers import pass_message, fail_message, skip_message
from helpers import clear_temporary_dir
from helpers import DummyModel

"""
The model is saved via PyTorch-Lightning's ModelCheckpoint 
callback method and contains all required information:
    - 16-bit scaling factor (if using 16-bit precision training)
    - Current epoch
    - Global step
    - LightningModule's state_dict
    - State of all optimizers
    - State of all learning rate schedulers
    - State of all callbacks
    - The hyperparameters used for that model if passed in as hparams (Argparse.Namespace)
    - State of Loops (if using Fault-Tolerant training)
"""


TEST_NAME = os.path.basename(__file__).split(".")[0]
TMP_DIR = "tmp"


def run_test(strategy, precision, gpus):

    # setup

    test_params = "s={}, p={}, g={}".format(
                strategy, precision, gpus)

    model = DummyModel()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath = TMP_DIR, 
            save_last = True, 
            every_n_epochs = 1)

    batch_limit = 2
    num_epochs = 1


    # training

    trainer = pl.Trainer(
        max_epochs = num_epochs,
        logger = False,
        strategy = strategy, 
        precision=precision, 
        gpus=gpus,
        limit_train_batches = batch_limit,
        limit_val_batches = batch_limit,
        enable_model_summary = False,
        enable_progress_bar = False,
        callbacks = [checkpoint_callback],
        )

    trainer.fit(model)


    # resume training

    model_ = DummyModel()

    trainer_ = pl.Trainer(
        max_epochs = num_epochs,
        logger = False,
        strategy = strategy, 
        precision=precision, 
        gpus=gpus,
        limit_train_batches = batch_limit,
        limit_val_batches = batch_limit,
        enable_model_summary = False,
        enable_progress_bar = False,
        enable_checkpointing = False,
        )

    passed = True
    try: # automatically restores model, epoch, step, LR schedulers, etc.
        trainer_.fit(
            model = model_, 
            ckpt_path = os.path.join(TMP_DIR, "last.ckpt")
        )
    except Exception as e:
        print(e)
        passed = False


    # print message

    if passed:
        pass_message(TEST_NAME, test_params)
        return 1
    else:
        fail_message(TEST_NAME, test_params)
        return 0



if __name__ == "__main__":

    list_strategy = ["ddp"]
    list_precision = [16, 32]
    list_gpus = [None, [0], [0,1]]

    total, passed = 0, 0

    for s, p, g in itertools.product(list_strategy, list_precision, list_gpus):
        clear_temporary_dir(TMP_DIR)
        passed += run_test(strategy=s, precision=p, gpus=g)
        total += 1

    clear_temporary_dir(TMP_DIR)

    str_passed = f"passed {passed}/{total}"
    if total == passed:
        pass_message(TEST_NAME, str_passed)
    else:
        fail_message(TEST_NAME, str_passed)
    
