from typing import Tuple, List, Union
from omegaconf import OmegaConf


def field_exists_and_is_not_null(config: OmegaConf, field: str) -> bool:
            if field in config:
                if not OmegaConf.is_none(config, field):
                    return True
            return False


