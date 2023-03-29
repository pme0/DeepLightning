import os
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    
    
class BoringModel(LightningModule):
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

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)
    
    
def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    model = BoringModel()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'TRAINABLE PARAMS: {params}')
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=True,
        strategy="deepspeed_stage_3",
        gpus=[0]
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    run()
