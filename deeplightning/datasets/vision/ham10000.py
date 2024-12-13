import os
from typing import Optional, Callable

from omegaconf import DictConfig
import pandas as pd
import lightning as pl
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from deeplightning import TASK_REGISTRY
from deeplightning.transforms.transforms import load_transforms
from deeplightning.utils.messages import info_message


def _extract_classes(
    df: pd.DataFrame, 
    labels_str: list[str], 
    label2index: dict,
) -> list[int]:
    """Extract the class indices."""
    classes = df.loc[:, labels_str].apply(lambda x: x.idxmax(), axis=1)
    classes = [label2index[k] for k in classes]
    return classes


def _extract_masks(
    df: pd.DataFrame, 
    root: str,
) -> list[str]:
    """Extract the mask paths."""
    masks = ["{}".format(
        os.path.join(root, "masks", f"{df.loc[i,'image']}_segmentation.png"))
        for i in range(df.shape[0])]
    return masks


def _trim_dataset(
    df: pd.DataFrame, 
    cfg: DictConfig,
) -> pd.DataFrame:
    """Trim the dataset to a specified number of samples."""
    trim = False
    if "dataset_size" in cfg.data:
        if cfg.data.dataset_size is not None:
            if cfg.data.dataset_size > 0:
                trim = True
    if trim:
        size_before = df.shape[0]
        df = df.iloc[:cfg.data.dataset_size, :]
        size_after = df.shape[0]
        info_message(
            f"Dataset trimmed from {size_before} to {size_after} "
            f"samples (cfg.data.dataset_size={cfg.data.dataset_size}).")
    return df


class HAM10000_dataset(Dataset):
    def __init__(self, 
        cfg: DictConfig,
        transforms: T.Compose | None = None,
        mask_transforms: T.Compose | None = None,
    ):
        self.task = cfg.task.name
        self.root = cfg.data.root
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.label2index = {"MEL":0, "NV":1, "BCC":2, "AKIEC":3, "BKL":4, "DF":5, "VASC":6}
        metadata = pd.read_csv(os.path.join(self.root, "GroundTruth.csv"))
        images = ["{}".format(
            os.path.join(self.root, "images", f"{metadata.loc[i,'image']}.jpg")) 
            for i in range(metadata.shape[0])
        ]
        class_labels_str = list(self.label2index.keys())
        self.data = pd.DataFrame()
        self.data["images"] = images
        if self.task == "ImageClassification":
            self.data["labels"] = _extract_classes(metadata, class_labels_str, self.label2index)
        elif self.task == "ImageSemanticSegmentation":
            self.data["masks"] = _extract_masks(metadata, self.root)

        # trim dataset (useful for debugging)
        self.data = _trim_dataset(self.data, cfg)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        
        # Process inputs (images)
        sample["inputs_paths"] = self.data.loc[idx, "images"]
        sample["inputs"] = Image.open(sample["inputs_paths"]).convert('RGB')

        if self.transforms:
            sample["inputs"] = self.transforms(sample["inputs"])
        
        # Process targets (labels or masks)
        if self.task == "ImageClassification":
            sample["labels"] = self.data.loc[idx, "labels"]
        elif self.task == "ImageSemanticSegmentation":
            sample["masks_paths"] = self.data.loc[idx, "masks"]
            sample["masks"] = Image.open(sample["masks_paths"])
            if self.mask_transforms:
                sample["masks"] = self.mask_transforms(sample["masks"])
                # squeeze the channel dimension
                sample["masks"] = sample["masks"].squeeze(0)

        return sample


class HAM10000(pl.LightningDataModule):
    """HAM10000 Dataset for Image Classification and Semantic Segmentation.
    It contains dermatoscopic images from different populations, acquired and 
    stored by different modalities. Cases include a representative collection 
    of all important diagnostic categories in the realm of pigmented lesions.

    Info:
        image size: 600 x 450 (width x height).
        image channels: 3.
        samples: 10,015 samples in total; no pre-existing split. 
        normalization: constants computed using deeplightning utils are
            mean=(0.7639, 0.5463, 0.5703) and std=(0.0870, 0.1155, 0.1295).
        classes: 2 for segmentation task, lesion and non-lesion regions; 7 for 
            classification task, see labels and descriptions in the table below
        |-------|-------------|------------------------------------------------|
        | label | no. samples | description                                    |
        |-------|-------------|------------------------------------------------|
        | MEL   | 1113        | Melanoma                                       |
        | NV    | 6705        | Melanocytic nevi                               |
        | BCC   | 514         | Basal cell carcinoma                           |
        | AKIEC | 327         | Actinic keratoses and intraepithelial          |
        |       |             |     carcinoma / Bowen's disease                |
        | BKL   | 1099        | Benign keratosis-like lesions (solar           |
        |       |             |     lentigines / seborrheic keratoses and      |
        |       |             |     lichen-planus like keratoses)              |
        | DF    | 115         | Dermatofibroma                                 |
        | VASC  | 142         | Vascular lesions (angiomas, angiokeratomas,    |
        |       |             |     pyogenic granulomas and hemorrhage)        |
        |-------|-------------|------------------------------------------------|
        
    References:
        Tschandl, P., Rosendahl, C., & Kittler, H. (2018). "The HAM10000 
            dataset, a large collection of multi-source dermatoscopic images 
            of common pigmented skin lesions". Scientific data, 5(1), 1-9.
        <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T>

    Args:
        cfg: configuration object.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        self.DATASET = "HAM10000"
        self.IMAGE_SIZE = (600, 450)  # (width,height)
        self.NUM_CHANNELS = 3
        self.NORMALIZATION = {
            "mean": [0.7639, 0.5463, 0.5703],
            "std": [0.0870, 0.1155, 0.1295],
        }

        self.validate_config_params()

        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")
        self.mask_transforms = load_transforms(cfg=cfg, subset="mask")

    def validate_config_params(self):
        if self.cfg.task.name == "ImageClassification":
            assert self.cfg.task.model.args.num_classes == 7
        elif self.cfg.task.name == "ImageSemanticSegmentation":
            assert self.cfg.task.model.args.num_classes == 2
        else:
            raise ValueError
        if "normalize" in self.cfg.data.train_transforms:
            assert self.cfg.data.train_transforms.normalize.mean == self.NORMALIZATION["mean"]
            assert self.cfg.data.train_transforms.normalize.std == self.NORMALIZATION["std"]
        if "normalize" in self.cfg.data.test_transforms:
            assert self.cfg.data.test_transforms.normalize.mean == self.NORMALIZATION["mean"]
            assert self.cfg.data.test_transforms.normalize.std == self.NORMALIZATION["std"]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage) -> None:
        # load dataset
        ds = HAM10000_dataset(
            cfg = self.cfg,
            transforms = self.test_transforms,
            mask_transforms = self.mask_transforms,
        )

        # split dataset
        self.train_ds, self.val_ds = random_split(ds, [0.8, 0.2])
        self.test_ds = self.val_ds

        # overwrite training transforms
        self.train_ds.dataset.transforms = self.train_transforms

        info_message("Training set size: {:,d}".format(len(self.train_ds)))
        info_message("Validation set size: {:,d}".format(len(self.val_ds)))
        info_message("Testing set size: {:,d}".format(len(self.test_ds)))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_ds, 
            batch_size = self.cfg.data.batch_size,
            shuffle = True,
            num_workers = self.cfg.data.num_workers,
            persistent_workers = self.cfg.data.persistent_workers,
            pin_memory = self.cfg.data.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_ds, 
            batch_size = self.cfg.data.batch_size,
            shuffle = False,
            num_workers = self.cfg.data.num_workers,
            persistent_workers = self.cfg.data.persistent_workers,
            pin_memory = self.cfg.data.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.test_ds, 
            batch_size = self.cfg.data.batch_size,
            shuffle = False,
            num_workers = self.cfg.data.num_workers,
            persistent_workers = self.cfg.data.persistent_workers,
            pin_memory = self.cfg.data.pin_memory,
        )
    