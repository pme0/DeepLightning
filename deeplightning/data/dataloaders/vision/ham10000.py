from typing import Union, Optional, Callable
import os
from omegaconf import OmegaConf
import pandas as pd
import io
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import lightning as pl
from PIL import Image

#from deeplightning.utils.io_local import read_image
from deeplightning.data.transforms.transforms import load_transforms
from deeplightning.data.transforms._resize import Resize
from deeplightning.utils.messages import info_message


def _extract_classes(metadata, class_labels_str, label2index):
    classes = metadata.loc[:, class_labels_str].apply(lambda x: x.idxmax(), axis=1)
    classes = [label2index[k] for k in classes]
    return classes


def _extract_masks(metadata, root):
    masks = ["{}".format(os.path.join(root, "masks", f"{metadata.loc[i,'image']}_segmentation.png")) for i in range(metadata.shape[0])]
    return masks


class HAM10000_dataset(Dataset):
    """HAM10000 Dataset for Image Classification and Semantic Segmentation.
    It contains dermatoscopic images from different populations, acquired and 
    stored by different modalities. Cases include a representative collection 
    of all important diagnostic categories in the realm of pigmented lesions.

    images and segmentation masks size: (width,height)=(600,450)
    normalization constants for images: mean=(?,) and std=(?,)
    number of samples: 10015
    number of image classes: 7
    number of segmentation classes: 7

    |-------|-------------|------------------------------------------------------------------------------------------------------------|
    | label | no. samples | description                                                                                                |
    |-------|-------------|------------------------------------------------------------------------------------------------------------|
    | MEL   | 1113        | melanoma                                                                                                   |
    | NV    | 6705        | melanocytic nevi                                                                                           |
    | BCC   | 514         | basal cell carcinoma                                                                                       |
    | AKIEC | 327         | actinic keratoses and intraepithelial carcinoma / Bowen's disease                                          |
    | BKL   | 1099        | benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)   |
    | DF    | 115         | dermatofibroma                                                                                             |
    | VASC  | 142         | vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)                            |
    |-------|-------------|------------------------------------------------------------------------------------------------------------|
        
    References:
        > "Human Against Machine with 10000 training images"
        > https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
        > Tschandl, P., Rosendahl, C., & Kittler, H. (2018). "The HAM10000 
            dataset, a large collection of multi-source dermatoscopic images 
            of common pigmented skin lesions". Scientific data, 5(1), 1-9.

    Args:
        cfg: configuration object
        transform: Transforms to be applied to images

    """
    def __init__(self, 
        task: str,
        root: str,
        transform: Union[Optional[Callable],None] = None,
        mask_transform: Union[Optional[Callable],None] = None,
    ):
        self.task = task
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.label2index = {"MEL":0, "NV":1, "BCC":2, "AKIEC":3, "BKL":4, "DF":5, "VASC":6}
        metadata = pd.read_csv(os.path.join(root, "GroundTruth.csv"))
        images = ["{}".format(os.path.join(root, "images", f"{metadata.loc[i,'image']}.jpg")) for i in range(metadata.shape[0])]
        
        class_labels_str = list(self.label2index.keys())
        self.data = pd.DataFrame()
        self.data["images"] = images
        if task == "ImageClassification":
            self.data["labels"] = _extract_classes(metadata, class_labels_str, self.label2index)
        elif task == "SemanticSegmentation":
            self.data["masks"] = _extract_masks(metadata, root)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        
        # Process inputs (images)
        sample["inputs_paths"] = self.data.loc[idx, "images"]
        sample["inputs"] = Image.open(sample["inputs_paths"]).convert('RGB')

        if self.transform:
            sample["inputs"] = self.transform(sample["inputs"])
        
        # Process targets (labels or masks)
        if self.task == "ImageClassification":
            sample["labels"] = self.data.loc[idx, "labels"]
        elif self.task == "SemanticSegmentation":
            sample["masks_paths"] = self.data.loc[idx, "masks"]
            sample["masks"] = Image.open(sample["masks_paths"])#.convert('RGB')
            if self.mask_transform:
                # mask must be resized and converted to tensor 
                # but no other transformations should be applied
                sample["masks"] = self.mask_transform(sample["masks"])
                # squeeze the channel dimension and convert to integer
                sample["masks"] = sample["masks"].squeeze(0).long()

        return sample
    

class HAM10000(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

        # check that config params are set correctly
        assert cfg.data.num_channels == 3
        if cfg.task == "ImageClassification":
            assert cfg.data.num_classes == 7
        elif cfg.task == "SemanticSegmentation":
            assert cfg.data.num_classes == 2
        else:
            raise ValueError

        # define data transforms
        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")
        self.mask_transform = T.Compose(T.ToTensor())
        resize = self.cfg.data.train_transforms.resize
        if resize is not None:
            self.mask_transform = T.Compose([Resize(resize), T.ToTensor()])

    def prepare_data(self) -> None:
        pass

    def setup(self, stage) -> None:
        ds = HAM10000_dataset(
            task = self.cfg.task,
            root = self.cfg.data.root,
            transform = self.test_transforms,
            mask_transform = self.mask_transform,
        )
        self.train_ds, self.val_ds = random_split(ds, [0.8, 0.2])
        self.test_ds = self.val_ds
        self.train_ds.dataset.transform = self.train_transforms  # overwrite

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