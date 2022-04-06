#!/bin/bash

conda activate deeplightning

pytest -q test_ckpt.py -r s

#python3 test_simple_trainer.py
#python3 test_ckpt_ds_consolidated.py
#python3 test_ckpt_ds_sharded.py
#python3 test_resume_ddp.py


#for f in test_*.py; do python3 "$f"; done