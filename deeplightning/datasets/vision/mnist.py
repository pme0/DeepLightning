from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
import lightning as pl

from deeplightning.datasets.vision import mnist_torchvision as mnist  # replaces `from torchvision.datasets import mnist``
from deeplightning.transforms import load_transforms
from deeplightning.utils.messages import info_message


class MNIST(pl.LightningDataModule):
    """MNIST data module.

    Args:
        cfg: configuration object.

    Info:
        image size: 28x28.
        image channels: 1.
        number of classes: 10, digits from 1 to 9.
        samples: 60,000 samples for training, to be split between training 
            samples and validation samples; 10,000 samples for testing.
        normalization: constants typically used in the literature for this 
            dataset are mean=(0.1307) and std=(0.3081).
    """
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

        self.DATASET = "MNIST"
        self.IMAGE_SIZE = (28, 28)  # (width,height)
        self.NUM_CHANNELS = 1
        self.NUM_CLASSES = 10
        self.NORMALIZATION_CONSTANTS = {
            "mean": [0.1307],
            "std": [0.3081]}
        
        self.validate_config_params(cfg)

        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")

    def validate_config_params(self, cfg):
        assert cfg.data.dataset == self.DATASET
        assert cfg.data.image_size[0] == self.IMAGE_SIZE[0]
        assert cfg.data.image_size[1] == self.IMAGE_SIZE[1]
        assert cfg.data.num_channels == self.NUM_CHANNELS
        assert cfg.data.num_classes == self.NUM_CLASSES

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