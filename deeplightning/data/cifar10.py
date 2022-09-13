from omegaconf import OmegaConf
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from deeplightning.utilities.messages import info_message
from deeplightning.data.transforms import get_transforms

class CIFAR10(pl.LightningDataModule):
    """ CIFAR10 dataset
    
    - classes: 10
    - training samples: 50,000
    - testing samples: 10,000
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

        #self.train_transform = get_transforms(cfg, "train_transforms")
        
        trfs = [transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]
        if "normalize" in cfg.data:
            if cfg.data.normalize:
                trfs.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        if "resize" in cfg.data:
            trfs.append(transforms.Resize(cfg.data.resize))
        
        self.transform = transforms.Compose(trfs)

    def prepare_data(self) -> None:
        datasets.CIFAR10(
            root = self.cfg.data.root, 
            train = True, 
            download = True)
        datasets.CIFAR10(
            root = self.cfg.data.root, 
            train = False, 
            download = True)

    def setup(self, stage) -> None:
        self.train_ds = datasets.CIFAR10(
            root = self.cfg.data.root, 
            train = True, 
            download = False, 
            transform = self.transform
        )  
        self.val_ds = datasets.CIFAR10(
            root = self.cfg.data.root, 
            train = False, 
            download = False, 
            transform = self.transform
        )      
        """ 
        The CIFAR10 dataset contains a training subset and a testing subset.
        Here we use the testing data in the validation dataloader.
        In case validation and testing dataloaders are required 
        (e.g. cross-validation), use the following:
        ```
        self.test_ds = CIFAR10(root = self.cfg.data.root, train = False, download = False, transform = self.transform)
        mnist_full = CIFAR10(root = self.cfg.data.root, train = True, download = False, transform = self.transform)
        self.train_ds, self.val_ds = random_split(mnist_full, [55000, 5000])
        ```
        """
        info_message("Training set size: {:,d}".format(len(self.train_ds)))
        info_message("Validation set size: {:,d}".format(len(self.val_ds)))
        #info_message("Testing set size: {:,d}".format(len(self.test_ds)))

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
        pass

    def predict_dataloader(self) -> DataLoader:
        pass
