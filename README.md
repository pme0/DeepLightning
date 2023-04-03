<h1 align="center">
  <b>Deep Lightning</b><br>
</h1>
<p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10-ff69b4" /></a>
    <a href= "https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0-orange" /></a>
    <a href= "https://www.pytorchlightning.ai"><img src="https://img.shields.io/badge/PyTorchLightning-2.0-yellow" /></a>
    <a href= "https://www.deepspeed.ai"><img src="https://img.shields.io/badge/DeepSpeed-0.5-blue" /></a>
</p>

**Deep Lightning** is a configuration-based wrapper for training Deep Learning models with focus on parallel training, cross-platform compatibility and reproducibility. The philosophy is simple: from configuration to trackable and reproducible deep learning.

After defining modules and configuration, training deep learning models is simple:
<p align="left">
  <img src="media/code.gif" alt=""  width="600" />
</p>

<!--
```python
from deeplightning.configure import load_config, init_module, init_trainer

# load configuration
cfg = load_config("cfg.yaml")

# load modules
model = init_module(cfg, "model")
data = init_module(cfg, "data")
trainer = init_trainer(cfg)

# train model
trainer.fit(model, data)
```
-->


### Contents
* [Overview](#overview)
* [Installation](#installation)
* [Usage](#usage)
  * [Run](#run)
  * [Configure](#configure)
  * [Customize](#customize)
* [Examples](#examples) 
* [Results](#results) 
* [Development](#development)
* [Further Reading](#further-reading)

# Overview

### Features
- Simplified trainer with **PyTorch-Lightning**
- Experiment tracking and logging with **Weights and Biases**
- Memory-efficinet parallel training with **DeepSpeed**
- Deployment (prediction API) with **Flask**
- Implementations of popular tasks/models with [**Examples**](https://github.com/pme0/DeepLightning/tree/master/examples)

> See [Development](#development) for a list of functionalities.

# Installation

Pre-requirement: Anaconda (installation instructions [here](https://docs.anaconda.com/anaconda/install)).

Clone repo:
```bash
git clone https://github.com/pme0/DeepLightning.git
cd DeepLightning
```

Create conda environment:
```bash
conda env create -f conda_env.yaml
conda activate deeplightning
```

# Usage

## Run

for model **training** use
```bash
python train.py --cfg configs/base.yaml
```
where `cfg` is the configuration YAML file;
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

### \*Logic

All config fields labelled `type` correspond to target classes. The format is `MODULE.CLASS` and the code will load class `CLASS` from `MODULE.py` (relative path). Note that `MODULE` can itself be composite, `X.Y.Z`, in which case the class `CLASS` will be loaded from `X/Y/Z.py`. 
For example, `model.optimizer.target` could be existing `deepspeed.ops.adam.FusedAdam` or user-defined in `losses.custom.MyLoss`.


### Details
- `data` requires 
    - `root` folder; 
    - `dataset` name;
    - `num_workers` in the dataloader; 
    - `batch_size`for each iteration; 
    - `module` requires:
        -  `type`, target class of type [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html) (takes the full config as input), in the format explained above;\*
- `model` requires:
    - `module` requires:
        -  `type`, target class of type [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) (takes the full config as input), in the format explained above;\*
    - `network`,`optimizer`,`scheduler`,`loss` require:
        - `type`, target class in the format explained above;\*
        - `params`, parameters for that class with matching keywords; include an empty `params` field if no parameters are required; 
        - note: `scheduler` requires additional `call` field with subfields `interval` (equal to `step` or `epoch`) and `frequency` (no. steps or epochs) for when to make a scheduler update;
- `engine` requires computational engine parameters:
    - `accelerator`, CPU (`cpu`) or GPU (`gpu`);
    - `strategy`, parallel backend {`deepspeed_stage_1`, `deepspeed_stage_2`, `deepspeed_stage_3`} for [DeepSpeed](https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#deepspeed) backend or `ddp` for [DataParallel](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.strategies.DDPStrategy.html#pytorch_lightning.strategies.DDPStrategy) backend;
    - `devices` is number of cores int (`2`) for `cpu`-accelerated training, or gpu ids list (`[0,1]`) for `gpu`-accelerated training;
    - `num_nodes`, the number of computational nodes;
    - `precision`, floating-point precision {`16`, `32`};
- `train` training parameters;
    - `num_epochs`, number of training epochs;
    - `val_every_n_epoch` how often to perform validation;
    - `grad_accum_every_n_batches` gradient accumulation;
    - `ckpt_resume_path` resume training from checkpoint;
    - `ckpt_monitor_metric` e.g. `val_acc`, used in `ModelCheckpoint` callback;
    - `ckpt_every_n_epochs` how often to perform checkpointing;
    - `ckpt_save_top_k` how many checkpoints to store;
    - `early_stop_metric` e.g. `val_acc`, used in `EarlyStopping` callback;

- `logger` logger parameters;
    - `name` logger name;
    - `project_name` project name;
    - `log_every_n_steps` logging frequency;
 
### Example
```python
modes: 
  train: true
  test: false
  
task: ImageClassification
  
data:
  root: /data
  dataset: MNIST
  module:
    target: deeplightning.data.dataloaders.mnist.MNIST
  num_workers: 4
  batch_size: 256
   
model:
  module:
    target: deeplightning.task.img_classif.ImageClassification
  network:
    target: deeplightning.model.cnn.SimpleCNN
    params: 
      num_classes: 10
      num_channels: 1
  optimizer:
    target: torch.optim.SGD
    params:
      lr: 0.01
      weight_decay: 0.01
      momentum: 0.9
  scheduler:
    target: torch.optim.lr_scheduler.ExponentialLR
    params:
      gamma: 0.99
    call:
      interval: "epoch"
      frequency: 1
  loss:
    target: torch.nn.CrossEntropyLoss
    params:

engine:
  accelerator: gpu
  strategy: deepspeed_stage_3
  devices: [0,1]
  num_nodes: 1
  precision: 32
  
train:
  num_epochs: 10
  val_every_n_epoch: 1
  grad_accum_from_epoch: 0
  grad_accum_every_n_batches: 32
  ckpt_resume_path: null
  ckpt_monitor_metric: val_acc  # used in `ModelCheckpoint` callback
  ckpt_every_n_epochs: 1
  ckpt_save_top_k: 1
  early_stop_metric: null  # used in `EarlyStopping` callback
  early_stop_delta: 0.001
  early_stop_patience: 3

test:
  ckpt_test_path: /PATH_TO_CKPT # used only when `modes.test=True`
  
logger:
  name: wandb
  project_name: testproject
  tags: ["_"] # cannot be empty
  notes: null
  log_every_n_steps: 10
```

### Customize

> Make sure you're familiar with the [configuration logic and details](#configure)!

Beyond changing parameters values in existing configs, you can customize the following according to your needs:
- **custom model**: put your model in `models/customnet.py`, and update the config field `model.network.target` and any required parameters to point to your new model;
- **custom task**: duplicate the task module `lightning/model/classification.py`, rename it `lightning/model/customtask.py`, make the required modifications to run your task, and update the config field `model.module.target` to point to your new task module;
- **custom dataset**: duplicate the data module `lightning/data/mnist.py`, rename it `lightning/data/customdataset.py`, make the required modifications to load your dataset, and update the config field `data.module.target` to point to your new data module;


# Examples

See [`examples`](https://github.com/pme0/DeepLightning/tree/master/examples) for details.

# Results

[results on acceleration, memory use, etc.]

# Development

### Functionalities
- [x] tracking logger (losses, learning rate, etc.)
- [x] artifact storing (config, image, figure - TODO histogram)
- [x] parallel training
  - [x] multi-gpu
  - [x] multi-node
  - [x] backend engines:
    - [x] ddp
    - [x] deepspeed_stage_1 
    - [x] deepspeed_stage_2
    - [ ] deepspeed_stage_3 (TODO resuming, sharded initialization)
- [x] 16-bit precision
- [x] periodic model checkpoints
- [ ] resume training from model checkpoint --- `deepspeed` untested [[docs](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed)] [[docs](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#collating-single-file-checkpoint-for-deepspeed-zero-stage-3)];
- [ ] sharded loading via LightningModule hook `configure_sharded_model(self):` [[docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#enabling-module-sharding-for-maximum-memory-efficiency)];
- [x] gradient accumulation
- [x] early stopping
- [x] prediction API [TODO: add batch support]
- [ ] multiple losses/optimizers e.g. GAN; [[docs](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=configure_optimizers#configure-optimizers)]; though deepspeed doesn't allow this atm "DeepSpeed currently only supports single optimizer, single scheduler within the training loop." [[docs](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed)]
- [x] reproducible examples
  - [x] image classification
  - [x] image reconstruction

### Notes
- :triangular_flag_on_post: on `mlflow==1.23.1`, initialization causes the folder `mlruns` to be created automatically, which is a nuisance if you want to log to a different folder. Mentioned in issue [#3400](https://github.com/mlflow/mlflow/issues/3400) and addressed in pull request [#3410](https://github.com/mlflow/mlflow/pull/3410) --- confirm on next release;
- :triangular_flag_on_post: on `deepspeed=0.5.10`, optimizer `deepspeed.ops.adam.FusedAdam` gives `AssertionError: CUDA_HOME does not exist, unable to compile CUDA op(s)`. Mentioned in issue [#1279](https://github.com/microsoft/DeepSpeed/issues/1279);
- :warning: effective batch size is `batch * num_gpus * num_nodes` [[docs](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#batch-size)] but huge batch size can cause convergence difficulties [[paper](https://arxiv.org/abs/1706.02677)];
- :warning: deepspeed single-file checkpointing requires caution [[docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#collating-single-file-checkpoint-for-deepspeed-zero-stage-3)] [[docs](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.plugins.training_type.DeepSpeedPlugin.html)]


# Further Reading

- **Pytorch-Lightning**
  - Lightning organises modules as hardware-agnostic and loop-less code; separated model and backend engine for scalable deep learning
  - :information_source: [website](https://lightning.ai/)  :floppy_disk: [github](https://github.com/Lightning-AI/lightning)
- **W&B** 
  - W&B tracks machine learning experiments.
  -  :information_source: [website](https://wandb.ai/site)  :floppy_disk: [github](https://github.com/wandb/wandb)
- **DeepSpeed** 
  - DeepSpeed is a distributed backend which reduces the training memory footprint with a Zero Redundancy Optimizer (ZeRO). It partitions model states and gradients to save memory, unlike traditional data parallelism where memory states are replicated across data-parallel processes. This allows training of large models with large batch sizes.
  - :information_source: [website](https://www.deepspeed.ai)  :floppy_disk: [github](https://github.com/microsoft/DeepSpeed)  :page_with_curl: [ZeRO-3](https://arxiv.org/abs/1910.02054)
- **Flask**
  - Flask is a server-side web framework that supports building and deploying web applications such as ML prediction APIs.
  - :information_source: [website](https://flask.palletsprojects.com)  :floppy_disk: [github](https://github.com/pallets/flask)


