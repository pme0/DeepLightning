from typing import Union, Optional, Callable
import os
from omegaconf import OmegaConf
import pandas as pd
import io
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as pl
from PIL import Image

#from deeplightning.utils.io_local import read_image
from deeplightning.data.transforms.transforms import load_transforms
from deeplightning.utils.messages import info_message


def _extract_classes(metadata, class_labels_str, label2index):
    classes = metadata.loc[:, class_labels_str].apply(lambda x: x.idxmax(), axis=1)
    classes = [label2index[k] for k in classes]
    return classes


def _extract_masks(metadata, root):
    masks = ["{}".format(os.path.join(root, "masks", f"{metadata.loc[i,'image']}_segmentation.png")) for i in range(metadata.shape[0])]
    return masks


class HAM10000_dataset(Dataset):
    """HAM10000 dataset.

    Arguments
    ---------
    cfg : configuration object
    transform : Transforms to be applied to images
    """

    def __init__(self, 
        cfg: OmegaConf, 
        transform: Union[Optional[Callable],None]=None
    ):
        assert cfg.task in ["ImageClassification", "SemanticSegmentation"]
        self.cfg = cfg
        self.transform = transform
        self.label2index = {"MEL":0, "NV":1, "BCC":2, "AKIEC":3, "BKL":4, "DF":5, "VASC":6}
        
        class_labels_str = list(self.label2index.keys())
        metadata = pd.read_csv(os.path.join(cfg.data.root, "GroundTruth.csv"))
        images = ["{}".format(os.path.join(cfg.data.root, "images", f"{metadata.loc[i,'image']}.jpg")) for i in range(metadata.shape[0])]
        
        self.data = pd.DataFrame()
        self.data["image"] = images
        if cfg.task == "ImageClassification":
            self.data["class"] = _extract_classes(metadata, class_labels_str, self.label2index)
        elif cfg.task == "SemanticSegmentation":
            self.data["mask"] = _extract_masks(metadata, cfg.data.root)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {"image": Image.open(self.data.loc[idx, "image"]).convert('RGB')}
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        if self.cfg.task == "ImageClassification":
            sample["class"]: self.data.loc[idx, "class"]
        elif self.task == "SemanticSegmentation":
            sample["mask"] = self.data.loc[idx, "mask"]
            if self.transform:
                sample["mask"] = self.transform(sample["mask"])

        return sample
    

class HAM10000(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

        # check that config params are set correctly
        assert cfg.data.num_channels == 3
        assert cfg.data.num_classes == 7

        # load data transformations/augmentations
        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage) -> None:
        ds = HAM10000_dataset(
            cfg = self.cfg,
            transform = self.train_transforms,
        )
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, [0.8, 0.1, 0.1])
        self.test_ds.transform = self.test_transforms

        info_message("Training set size: {:,d}".format(len(self.train_ds)))
        info_message("Validation set size: {:,d}".format(len(self.val_ds)))
        info_message("Testing set size: {:,d}".format(len(self.test_ds)))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_ds, 
            batch_size = self.cfg.data.batch_size,
            shuffle = True,
            num_workers = self.cfg.data.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_ds, 
            batch_size = self.cfg.data.batch_size,
            shuffle = False,
            num_workers = self.cfg.data.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.test_ds, 
            batch_size = self.cfg.data.batch_size,
            shuffle = False,
            num_workers = self.cfg.data.num_workers,
        )