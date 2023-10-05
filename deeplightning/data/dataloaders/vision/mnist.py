from omegaconf import OmegaConf
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, random_split
import lightning as pl

from deeplightning.data.transforms.transforms import load_transforms
from deeplightning.utils.messages import info_message


class MNIST(pl.LightningDataModule):
    """MNIST dataset. The dataset contains a training subset and a 
    testing subset. We split training subset into train and val samples

    - image size: 28x28
    - classes: 10
    - training subset: 60,000 samples (to be split between training 
        samples and validation samples)
    - testing subset: 10,000 samples
    - normalization constants typically used in the literature 
        for this  dataset are mean=(0.1307,) and std=(0.3081,).

    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

        # check that config params are set correctly
        assert cfg.data.image_size == 28
        assert cfg.data.num_channels == 1
        assert cfg.data.num_classes == 10

        # load data transformations/augmentations
        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")
        
        # Alternatively, the transforms may be hard-coded here,
        # though this is not recommended. E.g.
        # ```
        #   self.train_transforms = T.Compose([T.HorizontalFlip(0.5), T.Normalize((0.1307,), (0.3081,))])
        #   self.test_transforms = T.Compose([T.Normalize((0.1307,), (0.3081,))])
        # ```

    def prepare_data(self) -> None:
        mnist.MNIST(
            root = self.cfg.data.root, 
            train = True, 
            download = True)
        mnist.MNIST(
            root = self.cfg.data.root, 
            train = False, 
            download = True)

    def setup(self, stage) -> None:
        self.test_ds = mnist.MNIST(
            root = self.cfg.data.root, 
            train = False, 
            download = False, 
            transform = self.test_transforms
        )
        mnist_full = mnist.MNIST(
            root = self.cfg.data.root,
            train = True,
            download = False,
            transform = self.train_transforms
        )
        self.train_ds, self.val_ds = random_split(mnist_full, [55000, 5000])

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