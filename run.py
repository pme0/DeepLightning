import argparse

import hydra
from omegaconf import DictConfig
import wandb

from deeplightning.core.pipeline import DeepLightningPipeline
from deeplightning.utils.context import train_context, eval_context


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-name", 
    type=str, 
    help="Filename of YAML configuration file. Overwrites `config_name` in hydra.main()."
)
parser.add_argument(
    "--config-path", 
    type=str, 
    default="configs", 
    help="Directory of YAML configuration file. Overwrites `config_path` in hydra.main()."
)
args = parser.parse_args()


@hydra.main(
    version_base=None,
    config_path=args.config_path, 
    config_name=args.config_name,
)
def _main(cfg: DictConfig) -> None:
    """Main function running initializations, training and evaluation."""

    pipeline = DeepLightningPipeline(cfg)

    with train_context(cfg.engine.seed):
        if cfg.modes.train:
            pipeline.train()
  
    with eval_context(cfg.engine.seed):
        if cfg.modes.train:
            pipeline.eval("best")
        elif cfg.modes.test:
            pipeline.eval("config")

    wandb.finish()


if __name__ == "__main__":
    _main()