from contextlib import contextmanager
import sys

from lightning.pytorch import seed_everything
import torch
import wandb

from deeplightning.utils.messages import warning_message


@contextmanager
def train_context(seed: int):
    """Context manager for training."""
    torch.cuda.empty_cache()

    if seed:
        seed_everything(seed, workers=True)
    
    try:
        yield
    except KeyboardInterrupt as e:
        warning_message(f"Interrupted by user: {e}")
        sys.exit()
    finally:
        torch.cuda.empty_cache()


@contextmanager
def eval_context(seed: int):
    """Context manager for evaluation."""
    torch.cuda.empty_cache()

    if seed:
        seed_everything(seed, workers=True)
    
    try:
        yield
    except KeyboardInterrupt as e:
        warning_message(f"Interrupted by user: {e}")
        sys.exit()
    finally:
        torch.cuda.empty_cache()
