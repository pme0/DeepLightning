from omegaconf import OmegaConf



class neptureLogger():
    """ Neptune Logger.
    """
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        self.init_neptune()

    def init_neptune(self):
        pass
