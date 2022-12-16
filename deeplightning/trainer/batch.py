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

            if isinstance(batch, dict):
                if "paths" in batch and "images" in batch and "labels" in batch:
                    return batch
                else:
                    raise ValueError("batch dictionary should have keys ['paths', 'images', 'labels'].")

            if dataset in ["MNIST", "CIFAR10"]:
                # assumes dataloaders return tuple (images, labels)
                batch = {"paths": None, "images": batch[0], "labels": batch[1]}
                return batch
            else:
                raise NotImplementedError