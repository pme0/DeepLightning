import os
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import PIL

from deeplightning.utilities.messages import info_message, warning_message


class FSD_dataset(Dataset):
    def __init__(self, cfg: OmegaConf, subfolder: str, transform=None):
        """
        Args
        ----------
        :cfg: path to dataset main folder

        :subfolder: path to dataset subfolder (e.g. 'train' or 'test')
        
        :transform: data transforms
        """
        super(FSD_dataset, self).__init__()
        self.subfolder = subfolder
        self.transform = transform
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
        if self.transform is not None:
            images = self.transform(images)
        return {"paths": self.images[idx], 
                "images": images, 
                "labels": labels}


class FreeSpokenDigit(pl.LightningDataModule):
    """ FREE SPOKEN DIGIT dataset
    https://github.com/Jakobovski/free-spoken-digit-dataset
    
    - classes: 10
    - samples: 3,000
        - training samples: 2,700
        - testing samples: 300

    Preprocessing
    ----------
    After downloading the dataset and resources from the link above, 
    following preprocessing steps:
    ```
        cd fsd
        mkdir spectrograms
        mkdir training-spectrograms
        mkdir testing-spectrograms
        python spectrometer.py  # change paths inside this script
        python train-test-split.py
    ```
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        
        # train transformation

        train_trfs = [transforms.ToTensor()]

        if "normalize" in cfg.data.train_transforms:
            if cfg.data.train_transforms.normalize:
                train_trfs.append(transforms.Normalize(tuple(cfg.data.train_transforms.normalize.mean), tuple(cfg.data.train_transforms.normalize.stdev)))
        
        if "resize" in cfg.data.train_transforms:
            if cfg.data.train_transforms.resize is not None:
                train_trfs.append(transforms.Resize(cfg.data.train_transforms.resize))
        
        self.train_transforms = transforms.Compose(train_trfs)

        # test transformations

        test_trfs = [transforms.ToTensor()]

        if "normalize" in cfg.data.test_transforms:
            if cfg.data.test_transforms.normalize:
                test_trfs.append(transforms.Normalize(tuple(cfg.data.test_transforms.normalize.mean), tuple(cfg.data.test_transforms.normalize.stdev)))
        
        if "resize" in cfg.data.test_transforms:
            if cfg.data.test_transforms.resize is not None:
                test_trfs.append(transforms.Resize(cfg.data.test_transforms.resize))
        
        self.test_transforms = transforms.Compose(test_trfs)


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
            transform = self.train_transforms,
        )

        self.val_ds = FSD_dataset(
            cfg = self.cfg,
            subfolder = "testing-spectrograms",
            transform = self.test_transforms,
        )

        self.test_ds = self.val_ds

    
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


    def predict_dataloader(self) -> DataLoader:
        pass
