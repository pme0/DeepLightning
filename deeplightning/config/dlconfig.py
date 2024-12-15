import os

from colorama import Fore, Back, Style
from colorama import init as colorama_init
from omegaconf import OmegaConf, ListConfig, DictConfig


colorama_init()


class DeepLightningConfig(DictConfig):
    def __init__(self, cfg: DictConfig):
        super().__init__(content=cfg)
        self.resolve_config()

    def resolve_config(self) -> None:
        OmegaConf.resolve(self)
        self = self.expand_home_directories(self)

    def expand_home_directories(self, cfg: DictConfig) -> None:
        """Expand home directories specified with '~/' into absolute paths."""
        if isinstance(cfg, DictConfig):
            for key, value in cfg.items():
                cfg[key] = self.expand_home_directories(value)
        elif isinstance(cfg, ListConfig):
            for index, item in enumerate(cfg):
                cfg[index] = self.expand_home_directories(item)
        elif isinstance(cfg, str):
            if cfg.startswith("~/"):
                return os.path.expanduser(cfg)
        return cfg

    def log_config(self) -> None:
        if not isinstance(self, DictConfig):
            raise  TypeError(
                f"Config to be logger should be type 'DictConfig', "
                f"found "
            )
        
        filedir = self.logger.runtime.artifact_path
        filename = "cfg.yaml"  # wandb already saves some 'config.yaml'
        fp = os.path.join(filedir, filename)
        
        if not os.path.exists(filedir):
            os.makedirs(filedir, exist_ok=True)
        
        OmegaConf.save(self, f=fp)
 
    def print_config(self) -> None:
        ruler = "".join(["="]*60) + "\n"
        space = " " * 10
        msg = OmegaConf.to_yaml(self)
        msg = f"\n{ruler}{space}CONFIGURATION\n{ruler}{msg}{ruler}"
        print(Fore.CYAN + msg + Style.RESET_ALL, flush=True)