import torch
from torch.utils.data import DataLoader, Dataset
import lightning as pl


class DummyDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class DummyModel(pl.LightningModule):
    def __init__(self, cfg = None):
        super().__init__()
        self.cfg = cfg
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
