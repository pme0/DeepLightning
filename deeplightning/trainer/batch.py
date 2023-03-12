from typing import Any


def dictionarify_batch(batch: Any, dataset: str):
            """Convert batch to dictionary format with
            keys ["paths", "images", "labels"]

            Parameters
            ----------
            batch : batch object which is output by the LightningDataModule
                and is the input to `training_step()`, `validation_step()`, and
                `testing_step()`

            dataset : the dataset name
            """

            if dataset in ["MNIST", "CIFAR10"]:
                # assumes dataloaders return tuple (images, labels)
                batch = {"paths": None, "inputs": batch[0], "labels": batch[1]}
                return batch
            
            elif isinstance(batch, dict):
                if "paths" in batch and "inputs" in batch and "labels" in batch and len(batch.keys()) == 3:
                    return batch
                raise ValueError(f"batch dictionary keys unrecognised ({batch.keys()}).")

            else:
                raise NotImplementedError