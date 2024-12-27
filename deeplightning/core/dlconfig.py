import os
from colorama import Fore, Back, Style
from colorama import init as colorama_init
from dataclasses import dataclass, field, MISSING
from omegaconf import OmegaConf, ListConfig, DictConfig
from typing import Any, Optional

#from deeplightning.datasets import DATA_REGISTRY


colorama_init()


DATASETS = [
    "HAM10000",
    "CIFAR10",
    "MNIST",
]


@dataclass
class ModuleConfig:
    target: str
    args: Any = None


@dataclass
class TransformsConfig:
    train: Optional[dict[str, Any]] = field(default_factory=lambda: {})
    test: Optional[dict[str, Any]] = field(default_factory=lambda: {})
    mask: Optional[dict[str, Any]] = field(default_factory=lambda: {})


@dataclass
class DataConfig:
    dataset: str
    root: str
    batch_size: int
    debug_batch_size: int | None  # maximum number of samples, useful for debugging
    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    transforms: TransformsConfig
    module: ModuleConfig

    def __post_init__(self):

        if self.dataset not in DATASETS:
            datasets_str = "{" + ",".join(f'"{item}"' for item in DATASETS) + "}"
            raise ValueError(
                f"Dataset must be {datasets_str}, found {self.dataset}"
            )

        if self.debug_batch_size is not None:
            if self.debug_batch_size <= 0:
                raise ValueError(
                    f"Parameter 'debug_batch_size' must be None or greater than "
                    f"zero, found {self.debug_batch_size}."
                )
            

@dataclass
class TestConfig:
    active: bool
    ckpt_test_path: str | None  # only used when `stages.test.active` is true
    

@dataclass
class TrainConfig:
    active: bool
    num_epochs: int
    val_every_n_epoch: int
    grad_accum_from_epoch: int
    grad_accum_every_n_batches: int
    ckpt_resume_path: str | None
    ckpt_monitor_metric: str  # used in `ModelCheckpoint` callback
    ckpt_monitor_mode: str  # used in `ModelCheckpoint` callback
    ckpt_every_n_epochs: int
    ckpt_save_top_k: int
    early_stop_metric: str | None
    early_stop_delta: float
    early_stop_patience: int
    

@dataclass
class StagesConfig:
    train: TrainConfig
    test: TestConfig

    def __post_init__(self):
        pass


@dataclass
class ModelConfig:
    target: str
    args: dict


@dataclass
class OptimizerConfig:
    target: str
    args: Any


@dataclass
class SchedulerCall:
    interval: str
    frequency: int
    

@dataclass
class SchedulerConfig:
    target: str
    args: Any
    call: SchedulerCall


@dataclass
class LossConfig:
    target: str
    args: Any


@dataclass
class EngineConfig:
    accelerator: str
    strategy: str
    devices: Any  # ideally want `list[int] | str | int` but currently not supported ("omegaconf.errors.ConfigValueError: Unions of containers are not supported")
    num_nodes: int
    precision: int
    seed: int

    def __post_init__(self):
        pass


@dataclass
class LoggerRuntimeConfig:
    run_id: str | None = None
    run_name: str | None = None
    run_dir: str | None = None
    artifact_path: str | None = None
    

@dataclass
class LoggerConfig:
    provider: str
    project: str
    log_every_n_steps: int
    runtime: LoggerRuntimeConfig
    notes: str | None = None
    tags: list[str] = field(default_factory=lambda: ["none"])  # cannot be empty

    def __post_init__(self):
        pass


@dataclass
class MetricsConfig:
    train: Optional[list[str]] = field(default_factory=lambda: [])
    val: Optional[list[str]] = field(default_factory=lambda: [])
    test: Optional[list[str]] = field(default_factory=lambda: [])
    

@dataclass
class TaskConfig:
    name: str
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    metrics: MetricsConfig | None = None
    optionals: Optional[Any] | None = None

    def __post_init__(self):
        pass
    
    
@dataclass
class DeepLightningConfig:
    stages: StagesConfig = field(default=MISSING)
    data: DataConfig = field(default=MISSING)
    task: TaskConfig = field(default=MISSING)
    engine: EngineConfig = field(default=MISSING)
    logger: LoggerConfig = field(default=MISSING)

    def __post_init__(self):

        if isinstance(self.stages, StagesConfig):
            self.stages.__post_init__()
        if isinstance(self.data, DataConfig):
            self.data.__post_init__()
        if isinstance(self.task, TaskConfig):
            self.task.__post_init__()
        if isinstance(self.engine, EngineConfig):
            self.engine.__post_init__()
        if isinstance(self.logger, LoggerConfig):
            self.logger.__post_init__()
            
        print("\U0001f44d Configuration validated successfully \U0001f44d ")


    def log_config(self) -> None:
        if not isinstance(self, DeepLightningConfig):
            raise  TypeError(
                f"Config to be logger should be type 'DeepLightningConfig', "
                f"found {type(self)}"
            )
        
        filedir = self.logger.runtime.artifact_path
        filename = "cfg.yaml"  # wandb already saves some 'config.yaml'
        
        if not os.path.exists(filedir):
            os.makedirs(filedir, exist_ok=True)
        
        fp = os.path.join(filedir, filename)
        OmegaConf.save(self, f=fp)
 

    def print_config(self) -> None:
        ruler = "".join(["="]*60) + "\n"
        space = " " * 10
        msg = OmegaConf.to_yaml(self)
        msg = f"{ruler}{space}CONFIGURATION\n{ruler}{msg}{ruler}"
        print(Fore.CYAN + msg + Style.RESET_ALL, flush=True)


def expand_home_directories(cfg: DictConfig) -> DictConfig:
    """Expand directories starting with '~/' into absolute paths."""
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            cfg[key] = expand_home_directories(value)
    elif isinstance(cfg, ListConfig):
        for index, item in enumerate(cfg):
            cfg[index] = expand_home_directories(item)
    elif isinstance(cfg, str):
        if cfg.startswith("~/"):
            return os.path.expanduser(cfg)
    return cfg



def dataclass_to_dict(obj):
    """Recursively convert dataclass to dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {
            field.name: dataclass_to_dict(getattr(obj, field.name)) 
            for field in obj.__dataclass_fields__.values()
        }
    elif isinstance(obj, list):
        return [
            dataclass_to_dict(item) 
            if hasattr(item, "__dataclass_fields__") 
            else item 
            for item in obj
        ]
    else:
        return obj


def reload_config(cfg: DictConfig) -> DeepLightningConfig:
    OmegaConf.resolve(cfg)
    cfg = expand_home_directories(cfg)
    cfg = OmegaConf.merge(OmegaConf.structured(DeepLightningConfig), cfg)
    cfg = OmegaConf.to_object(cfg)
    return cfg
