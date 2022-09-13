#!/bin/bash

conda activate deeplightning
pip install pytest

#pytest -q test_initialization.py -r s
#pytest -q test_trainer.py -r s
pytest -q test_checkpoint.py -r s


#for f in test_*.py; do pytest -q "$f" -r s; done