from typing import Any


def dictionarify_batch(batch: Any, dataset: str) -> dict:
    """Convert batch to dictionary format.
    
    Typically keys in this dictionary would be `inputs`, `targets`
    or `paths`, but these may be anything. Note that the trainer 
    hooks must use these same keys when accessing elements from 
    the batch dictionary.

    Args:
    batch: batch object which is output by the LightningDataModule
        and is the input to `training_step()`, `validation_step()`
        and `testing_step()`.
    dataset: the dataset name, to make pre-existing datasets conform
        with the batch dictionary convention.
    """
    if dataset in ["MNIST", "CIFAR10"]:
        batch = {"inputs_paths": None, "inputs": batch[0], "targets": batch[1]}
        return batch
    elif isinstance(batch, dict):
        return batch
    else:
        raise NotImplementedError