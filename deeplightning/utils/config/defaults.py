__ConfigGroups__ = [
    "task_defaults",
    "data_defaults", 
    "model_defaults", 
    "engine_defaults", 
    "train_defaults", 
    "logger_defaults"
    ]

task_defaults = \
    """
    task: null
    """

data_defaults = \
    """
    data:
        root: null
        dataset: null
        num_workers: 4
        batch_size: 64
        module:
            target: deeplightning.datasets.mnist.MNIST
    """

model_defaults = \
    """
    model:
        module:
            type: null
        network:
            type: null
            args: 
                num_classes: null
                num_channels: null
        optimizer:
            type: null
            args:
                lr: null
        scheduler:
            type: null
            args:
            call:
                interval: epoch
                frequency: 1
        loss:
            type: null
            args:
    """

engine_defaults = \
    """
    engine:
        backend: null
        gpus: null
        num_nodes: 1
        precision: 32
    """

train_defaults = \
    """
    train:
        num_epochs: null
        val_every_n_epoch: 1
        grad_accum_from_epoch: 0
        grad_accum_every_n_batches: 1
        ckpt_resume_path: null
        ckpt_every_n_epochs: 1
        early_stop_metric: null
        early_stop_delta: 0.001
        early_stop_patience: 3
    """

logger_defaults = \
    """
    logger:
        type: pytorch_lightning.loggers.MLFlowLogger
        args:
            experiment_name: Default
            tracking_uri: mlruns
        log_every_n_steps: 10
    """