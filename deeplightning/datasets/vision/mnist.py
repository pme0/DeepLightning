from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, random_split
import lightning as pl

from deeplightning import DATA_REGISTRY
from deeplightning.core.dlconfig import DeepLightningConfig
from deeplightning.datasets.vision import mnist_torchvision
from deeplightning.transforms.transforms import load_transforms
from deeplightning.utils.data import do_trim
from deeplightning.utils.messages import info_message
from deeplightning.datasets.info import DATASET_INFO


class MNIST(pl.LightningDataModule):
    """MNIST data module.

    Args:
        cfg: configuration object.

    Info:
        image size: 28x28.
        image channels: 1.
        number of classes: 10, digits from 0 to 9, with class label equal to
            the digit number i.e. samples of ones have label '1' (int).
        samples: 60,000 samples for training, to be split between training 
            samples and validation samples; 10,000 samples for testing.
        normalization: constants typically used in the literature for this 
            dataset are mean=(0.1307) and std=(0.3081).
    """
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.DATASET_INFO = DATASET_INFO[cfg.data.dataset]

        self._validate_config_params()

        self.train_transforms = load_transforms(cfg=cfg, subset="train")
        self.test_transforms = load_transforms(cfg=cfg, subset="test")

    def _trim_mnist_dataset(self, ds):
        if do_trim(self.cfg):
            size_before = len(ds)
            ds = Subset(ds, range(self.cfg.data.debug_batch_size))
            size_after = len(ds)
            info_message(
                f"Dataset trimmed from {size_before} to {size_after} samples."
            )
            return ds
        return ds
    
    def _split_mnist_dataset(self, ds):
        base_split = [55000, 5000]
        if do_trim(self.cfg):
            prop_train = base_split[0] / sum(base_split)
            debug_samples = self.cfg.data.debug_batch_size
            debug_split = [
                int(debug_samples * prop_train)+1, 
                int(debug_samples * (1 - prop_train)),
            ]
            assert sum(debug_split) == debug_samples
            return random_split(ds, debug_split)
        return random_split(ds, base_split)

    
    def _validate_config_params(self):
        pass

    def prepare_data(self) -> None:
        mnist_torchvision.MNIST(
            root = self.cfg.data.root, 
            train = True, 
            download = True)
        mnist_torchvision.MNIST(
            root = self.cfg.data.root, 
            train = False, 
            download = True)

    def setup(self, stage) -> None:
        self.test_ds = mnist_torchvision.MNIST(
            root = self.cfg.data.root, 
            train = False, 
            download = False, 
            transform = self.test_transforms
        )
        mnist_full = mnist_torchvision.MNIST(
            root = self.cfg.data.root,
            train = True,
            download = False,
            transform = self.train_transforms
        )

        mnist_full = self._trim_mnist_dataset(mnist_full)
        self.train_ds, self.val_ds = self._split_mnist_dataset(mnist_full)
        
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
    

@DATA_REGISTRY.register_element()
def mnist(cfg: DeepLightningConfig) -> MNIST:
    return MNIST(cfg)