import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import shutil
import os
from colorama import Fore, Back, Style
from colorama import init as colorama_init
colorama_init()


def test_status_checker(test_name, test_params, passed, return_str=False):
    assert isinstance(passed, bool)
    if passed:
        return 1, pass_message(test_name, test_params, return_str)
    else:
        return 0, fail_message(test_name, test_params, return_str)


def print_test_report(test_name, passed, total, report):
    print(Fore.BLUE + "REPORT:" + Style.RESET_ALL)
    for x in report:
        print(x)
    str_passed = f"passed {passed}/{total}"
    if total == passed:
        pass_message(test_name, str_passed)
    else:
        fail_message(test_name, str_passed)


def get_rank():
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/rank_zero.py
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


@rank_zero_only
def assert_shards(ckpt_path, gpus, file, test_params):
    with open(os.path.join(ckpt_path, "latest"), "r") as f:
        latest = f.readline().strip()
    ckpt_path_contents = os.listdir(os.path.join(ckpt_path, latest))
    num_shards = sum([1 for x in ckpt_path_contents if x.endswith("model_states.pt")])
    num_gpus = len(gpus)
    if num_gpus != num_shards: 
        print(
            Fore.RED +
            "[{}][{}] Improper sharding: number of shards ({}) does match number of "
            "gpus in use ({}). Try increasing model parameters.".format(
                file, test_params, num_shards, num_gpus) + 
            Style.RESET_ALL, flush=True)
        return 0


@rank_zero_only
def rank_zero_pass_message(file, test):
    pass_message(file, test)
    

@rank_zero_only
def rank_zero_fail_message(file, test):
   fail_message(file, test)
    

@rank_zero_only
def rank_zero_skip_message(file, test):
    skip_message(file, test)


def pass_message(file, test, return_str = False):
    s = Fore.GREEN + f"[{file}][{test}] Passed" + Style.RESET_ALL
    if return_str:
        return s
    print(s, flush=True)


def skip_message(file, test, return_str = False):
    s = Fore.YELLOW + f"[{file}][{test}] Skipped" + Style.RESET_ALL
    if return_str:
        return s
    print(s, flush=True)


def fail_message(file, test, return_str = False):
    s = Fore.RED + f"[{file}][{test}] Failed" + Style.RESET_ALL
    if return_str:
        return s
    print(s, flush=True)


@rank_zero_only
def rank_zero_clear_temporary_dir(dir="tmp"):
    clear_temporary_dir(dir="tmp")


def clear_temporary_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)


class DummyDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(DummyDataset(32, 64), batch_size=2, num_workers=4)

    def val_dataloader(self):
        return DataLoader(DummyDataset(32, 64), batch_size=2, num_workers=4)

    def test_dataloader(self):
        return DataLoader(DummyDataset(32, 64), batch_size=2, num_workers=4)

        
class LargeDummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1000, 100)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(DummyDataset(1000, 64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(DummyDataset(1000, 64), batch_size=2)

    def test_dataloader(self):
        return DataLoader(DummyDataset(1000, 64), batch_size=2)
