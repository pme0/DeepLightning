__config_groups__ = [
    "data_defaults", 
    "model_defaults", 
    "engine_defaults", 
    "train_defaults", 
    "logger_defaults"
    ]

data_defaults = \
    """
    data:
        root: None
        dataset: None
        num_workers: 4
        batch_size: 64
        module:
            type: deeplightning.data.mnist.MNIST
    """

model_defaults = \
    """
    model:
        module:
            type: None
        network:
            type: None
            params: 
                num_classes: None
                num_channels: None
        optimizer:
            type: None
            params:
                lr: None
        scheduler:
            type: None
            params:
            call:
                interval: epoch
                frequency: 1
        loss:
            type: None
            params:
    """

engine_defaults = \
    """
    engine:
        backend: None
        gpus: null
        num_nodes: 1
        precision: 32
    """

train_defaults = \
    """
    train:
        num_epochs: None
        val_every_n_epoch: 1
        grad_accum_from_epoch: 0
        grad_accum_every_n_batches: 1
        ckpt_resume_path: null
        ckpt_every_n_epochs: 1
        early_stop_metric: None
        early_stop_delta: 0.001
        early_stop_patience: 3
    """

logger_defaults = \
    """
    logger:
        type: pytorch_lightning.loggers.MLFlowLogger
        params:
            experiment_name: Default
            tracking_uri: mlruns
        log_every_n_steps: 10
    """