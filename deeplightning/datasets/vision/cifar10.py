from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10 as torchvisionCIFAR10
from torch.utils.data import DataLoader, random_split
import lightning as pl

from deeplightning.utils.messages import info_message
from deeplightning.transforms import load_transforms


class CIFAR10(pl.LightningDataModule):
    """CIFAR10 dataset. The dataset contains a training subset and a 
    testing subset. We split training subset into train and val samples
    
    - image size: 32x32
    - classes: 10
    - training subset: 50,000 samples (to be split between training 
        samples and validation samples)
    - testing subset: 10,000 samples
    - normalization constants typically used in the literature for 
        this dataset are mean=(0.4914, 0.4822, 0.4465) and std=(0.2023, 0.1994, 0.2010)
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")

    def prepare_data(self) -> None:
        torchvisionCIFAR10(
            root = self.cfg.data.root, 
            train = True, 
            download = True)
        torchvisionCIFAR10(
            root = self.cfg.data.root, 
            train = False, 
            download = True)

    def setup(self, stage) -> None:
        self.test_ds = torchvisionCIFAR10(
            root = self.cfg.data.root, 
            train = False, 
            download = False, 
            transform = self.test_transforms,
        )      
        cifar10_train_subset = torchvisionCIFAR10(
            root = self.cfg.data.root, 
            train = True, 
            download = False, 
            transform = self.train_transforms
        )
        self.train_ds, self.val_ds = random_split(cifar10_train_subset, [45000, 5000])

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