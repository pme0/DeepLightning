def init_model(config: OmegaConf) -> LightningModule:
    """ Initialize LightningModule
    """
    s = config.model.module
    return init_module(short_config = s, config = config)
