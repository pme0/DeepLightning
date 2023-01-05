#!/bin/bash

# before running the script make sure to activate `deeplightning`
# environment and install `pytest` library
#conda activate deeplightning
#pip install pytest

# run `pytest --help` to see argument options

# the following runs `pytest` on all files in the current directory
pytest .  -q  -r A  --disable-pytest-warnings

#pytest test_trainer.py     -q  -r A  --disable-pytest-warnings
#pytest test_checkpoint.py  -q  -r A  --disable-pytest-warnings
#pytest test_transforms.py  -q  -r A  --disable-pytest-warnings
