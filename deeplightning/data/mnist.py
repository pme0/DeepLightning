from omegaconf import OmegaConf
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from deeplightning.utilities.messages import info_message


class MNIST(pl.LightningDataModule):
    """ MNIST dataset
    
    - classes: 10
    - training samples: 60,000
    - testing samples: 10,000
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.dataset = "MNIST"
        trfs = [transforms.ToTensor()]
        if "normalize" in cfg.data:
            if cfg.data.normalize:
                trfs.append(transforms.Normalize((0.1307,), (0.3081,)))
        if "resize" in cfg.data:
            if cfg.data.resize is not None:
                trfs.append(transforms.Resize(cfg.data.resize))
        self.transform = transforms.Compose(trfs)

    def prepare_data(self) -> None:
        datasets.MNIST(
            root = self.cfg.data.root, 
            train = True, 
            download = True)
        datasets.MNIST(
            root = self.cfg.data.root, 
            train = False, 
            download = True)

    def setup(self, stage) -> None:
        self.train_ds = datasets.MNIST(
            root = self.cfg.data.root, 
            train = True, 
            download = False, 
            transform = self.transform
        )
        self.val_ds = datasets.MNIST(
            root = self.cfg.data.root, 
            train = False, 
            download = False, 
            transform = self.transform
        )
        """ 
        The MNIST dataset contains a training subset and a testing subset.
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
        #utils.info_message("Testing set size: {:,d}".format(len(self.test_ds)))

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
