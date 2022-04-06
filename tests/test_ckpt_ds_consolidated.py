import itertools
import os
import sys
sys.path.insert(0, "..")

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import pytorch_lightning as pl

from helpers import DummyModel, LargeDummyModel
from helpers import (
    get_rank,
    assert_shards,
    rank_zero_clear_temporary_dir,
    rank_zero_pass_message,
    rank_zero_fail_message,
    pass_message,
    fail_message,
)


TESTNAME = os.path.basename(__file__).split(".")[0]
BATCH_LIMIT = 2
NUM_EPOCHS = 1
DIM_IN = 100
DIM_OUT = 10


def run_test(strategy, precision, gpus):
    """
    For sharded models, model.state_dict() does not give 
    the weights. Seeding the runs and comparing the saved 
    weights with the ground truth - obtained by running 
    the experiment on 1-gpu/cpu - does not work either.
    """

    seed_everything(1001)

    test_params = "strategy={}, precision={}, gpus={}".format(
                strategy, precision, gpus)

    # initialize a large model to induce sharding
    model = LargeDummyModel()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[{}] Trainable parameters: {:,d}".format(TESTNAME, trainable_params), flush=True)

    trainer = pl.Trainer(
        max_epochs = NUM_EPOCHS,
        logger = False,
        strategy = strategy, 
        precision=precision, 
        gpus=gpus,
        limit_train_batches = BATCH_LIMIT,
        limit_val_batches = BATCH_LIMIT,
        enable_model_summary = False,
        enable_progress_bar = False,
        callbacks = [pl.callbacks.ModelCheckpoint(dirpath="tmp", every_n_epochs=1)],
        )

    trainer.fit(model)

    print('RANKKKKKKKKKKKKKKK', get_rank())

    if True:
        # NOTE: deepspeed saves to folder rather than file 
        # and loading finds the file given the folder
        ckpt_path = "tmp/epoch={}-step={}.ckpt".format(
                NUM_EPOCHS-1, BATCH_LIMIT-1)

        status = assert_shards(ckpt_path, gpus, TESTNAME, test_params)
        if status == 0:
            return 0

        # consolidated sharded checkpoints
        consolidated_ckpt = "tmp/consolidated_state_dict.pt"
        convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_dir = ckpt_path, 
            output_file = consolidated_ckpt)

        # load model from consolidated checkpoint
        loaded_model = LargeDummyModel.load_from_checkpoint(
            consolidated_ckpt)

        if get_rank() == 0:
            print('MODEL', loaded_model.state_dict()['layer.weight'], flush=True)

        # compare weights for all layers
        passed = True
        for key in model.state_dict():
            if (model.state_dict()[key] != loaded_model.state_dict()[key]).sum().item() > 0:
                #print(F"KEY {key}", flush=True)
                #print('MODEL', model.state_dict()[key], flush=True)
                #print('LOADED MODEL', loaded_model.state_dict()[key], flush=True)
                passed = False
                
        # print message
        if passed:
            pass_message(TESTNAME, test_params)
            return 1
        else:
            fail_message(TESTNAME, test_params)
            return 0



if __name__ == "__main__":

    list_strategy = ['deepspeed_stage_3'] #["deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"]
    list_precision = [32] #[16, 32]
    list_gpus = [[0]] #[[0], [0,1]]

    total, passed = 0, 0

    for s, p, g in itertools.product(list_strategy, list_precision, list_gpus):
        #rank_zero_clear_temporary_dir()
        passed += run_test(strategy=s, precision=p, gpus=g)
        total += 1

    rank_zero_clear_temporary_dir()

    str_passed = f"passed {passed}/{total}"
    if total == passed:
        rank_zero_pass_message(TESTNAME, str_passed)
    else:
        rank_zero_fail_message(TESTNAME, str_passed)
    
