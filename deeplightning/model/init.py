def init_model(cfg: OmegaConf) -> LightningModule:
    """ Initialize LightningModule
    """
    s = cfg.model.module
    return init_module(short_cfg = s, cfg = cfg)
