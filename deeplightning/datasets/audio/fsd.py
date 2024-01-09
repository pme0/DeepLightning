import os
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as pl
import PIL

from deeplightning.utils.messages import info_message, warning_message
from deeplightning.transforms import load_transforms


class FSD_dataset(Dataset):
    """Free Spoken Digit dataset.

    Preprocessing:
        After downloading the dataset and code from 
        https://github.com/Jakobovski/free-spoken-digit-dataset
        follow the preprocessing steps to obtain the spectrograms:
        ```
            cd fsd
            mkdir spectrograms
            mkdir training-spectrograms
            mkdir testing-spectrograms
            python spectrometer.py  # change paths inside this script
            python train-test-split.py
        ```

    Args:
        cfg: configuration.
        subfolder: path to dataset subfolder (e.g. 'train' or 'test').
        transforms: composition of torchvision data transforms.
    """
    def __init__(self, cfg: OmegaConf, subfolder: str, transforms=None):
        super(FSD_dataset, self).__init__()
        self.subfolder = subfolder
        self.transforms = transforms
        self.subfolder_path = os.path.join(cfg.data.root, subfolder)
        extensions = tuple([".png"])  # valid extensions
        self.images = [os.path.join(self.subfolder_path, x) for x in os.listdir(self.subfolder_path) if x.endswith(extensions)]
        self.labels = [self.extract_class_from_filename(x) for x in self.images]

    def extract_class_from_filename(self, filename: str):
        return int(filename.split('/')[-1].split('.')[0].split('_')[0])
        
    def pil_loader(self, path: str) -> PIL.Image.Image:
        with open(path, "rb") as f:
            img = PIL.Image.open(f)
            return img.convert("RGB").convert("L")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        images = self.pil_loader(self.images[idx])
        if self.transforms is not None:
            images = self.transforms(images)
        return {"paths": self.images[idx], 
                "images": images, 
                "labels": labels}


class FreeSpokenDigit(pl.LightningDataModule):
    """Lightning Data Module for Free Spoken Digit dataset. 
    See https://github.com/Jakobovski/free-spoken-digit-dataset
    
    Info:
        classes: 10.
        samples: 3,000 total; 2,700 training, 300 testing.

    Args:
        cfg: configuration.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        
        # set dataset parameters
        self.DATASET = "FSD"
        self.IMAGE_SIZE = (64, 64)  # (width,height)
        self.NUM_CHANNELS = 1
        self.NUM_CLASSES = 10
        #self.NORMALIZATION = {"mean": [], "std": []}
        self.validate_config_params(cfg)

        # load data transformations/augmentations
        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")

    def validate_config_params(self, cfg):
        assert cfg.data.dataset == self.DATASET
        assert cfg.data.image_size[0] == self.IMAGE_SIZE[0]
        assert cfg.data.image_size[1] == self.IMAGE_SIZE[1]
        assert cfg.data.num_channels == self.NUM_CHANNELS
        assert cfg.data.num_classes == self.NUM_CLASSES
        
    def prepare_data(self) -> None:
        pass

    def setup(self, stage) -> None:
        """ 
        The FSD dataset contains a training subset and a testing subset.
        Here we use the testing datset for both 'val' and 'test' subsets.
        """

        self.train_ds = FSD_dataset(
            cfg = self.cfg,
            subfolder = "training-spectrograms",
            transforms = self.train_transforms,
        )

        self.val_ds = FSD_dataset(
            cfg = self.cfg,
            subfolder = "testing-spectrograms",
            transforms = self.test_transforms,
        )

        # TODO currently testing is the same as validation;
        # split `testing-spectrograms` into two subsets
        self.test_ds = FSD_dataset(
            cfg = self.cfg,
            subfolder = "testing-spectrograms",
            transforms = self.test_transforms,
        )
    
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


    #def predict_dataloader(self) -> DataLoader:
    #    pass
