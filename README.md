<h1 align="center">
  <b>Deep Lightning</b><br>
</h1>
<p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10-brightgreen" /></a>
    <a href= "https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0-yellow" /></a>
    <a href= "https://www.pytorchlightning.ai"><img src="https://img.shields.io/badge/Lightning-2.0-orange" /></a>
</p>

**Deep Lightning** is a configuration-based wrapper for training Deep Learning models with focus on parallel training, cross-platform compatibility and reproducibility. The philosophy is simple: from configuration to trackable and reproducible deep learning.

After defining modules and configuration, training deep learning models is simple:
<p align="left">
  <img src="media/code.gif" alt=""  width="600" />
</p>

### Contents
* [Overview](#overview)
* [Installation](#installation)
* [Usage](#usage)
  * [Run](#run)
  * [Configure](#configure)
  * [Customize](#customize)
* [Examples](#examples)

# Overview

### Features
- Simplified trainer with **PyTorch-Lightning**
- Experiment tracking and logging with **Weights and Biases**
- Memory-efficinet parallel training with **DeepSpeed**
- Deployment (prediction API) with **Flask**
- Implementations of popular tasks/models with [**Examples**](https://github.com/pme0/DeepLightning/tree/master/examples)

# Installation

Pre-requirement: Anaconda (installation instructions [here](https://docs.anaconda.com/anaconda/install)).

Clone repo:
```bash
git clone https://github.com/pme0/DeepLightning.git
cd DeepLightning
```

Create conda environment:
```bash
conda env create -f env.yaml
conda activate dl
```

# Usage

## Run

for model **training** use
```bash
python run.py \
  --config-name CONFIG_NAME.yaml \
  --config-path CONFIG_PATH.yaml \
```
where `config-path` defaults to `configs/` and `config-name` is the config filename;
To create your own config follow the [Configuration guidelines](#configure) or see [Examples](#examples).

**2. Monitor the training progress:**

When a training run has been initiated, a link will be displayed in the terminal; clicking it will open the Weights & Biases web interface. There you will be able to monitor the relevant metrics during training/testing and compare multiple runs:

<img src="media/wandb.png" width="700">

**3. Deploy the model:**
```bash
./deploy.sh <artifact-storage-path>

# Example:
# ./deploy.sh /mlruns/0/6ff30d9bc5b74c019071d575fec86a19/artifacts
```
- `artifact-storage-path` is the path where artifacts were stored during training, which contains the train config (`cfg.yaml`) and model checkpoint (`last.ckpt`);

**4. Predict using the API:**
```bash
./predict.sh <image>

# Example:
# ./predict.sh image.jpg
```
- `image` is the path to the image to be predicted;

## Configure

### Logic
All config fields labelled `type` correspond to target classes. The format is `MODULE.CLASS` and the code will load class `CLASS` from `MODULE.py` (relative path). Note that `MODULE` can itself be composite, `X.Y.Z`, in which case the class `CLASS` will be loaded from `X/Y/Z.py`. 
For example, `model.optimizer.target` could be existing `deepspeed.ops.adam.FusedAdam` or user-defined in `losses.custom.MyLoss`.
 
Example:
```yaml
model:
  module:
    target: deeplightning.tasks.vision.classification.ImageClassificationTask
  network:
    target: deeplightning.models.cnn.CNN
    args: 
      num_classes: 10
      num_channels: 1
```

### Customize

> Make sure you're familiar with the [configuration logic](#logic).

Beyond changing parameters values in existing configs, you can customize the following according to your needs:
- **custom model**: put your model in `models/customnet.py`, and update the config field `task.model.target` and any required parameters to point to your new model;
- **custom task**: duplicate the task module `lightning/model/classification.py`, rename it `lightning/model/customtask.py`, make the required modifications to run your task, and update the config field `model.module.target` to point to your new task module;
- **custom dataset**: duplicate the data module `lightning/data/mnist.py`, rename it `lightning/data/customdataset.py`, make the required modifications to load your dataset, and update the config field `data.module.target` to point to your new data module;


# Examples

See [`examples`](https://github.com/pme0/DeepLightning/tree/master/examples) for details.

