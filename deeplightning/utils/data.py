from deeplightning.core.dlconfig import DeepLightningConfig


def do_trim(cfg: DeepLightningConfig) -> bool:
    """Whether the dataset should be trimmed based on config parameters."""
    if cfg.data.debug_batch_size is not None:
        return True
    return False