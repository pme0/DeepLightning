# Test

## Installation
Activate environment and install `pytest` library
```bash
conda activate dl
pip install pytest
```

## Usage
From the main project directory run
```bash
pytest tests/ -q -r A --disable-pytest-warnings
```
for all tests, or 
```bash
pytest tests/testA.py -q -r A --disable-pytest-warnings
```
for a specific test.

Use `pytest --help` to see argument options.


Some debugging tools are in `tests/debug/` e.g.
```bash
python -m tests.debug.registries
```