from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader


def compute_dataset_mean_and_stdev(
    dataloader: DataLoader,
    batch_key: str,
    data_type: str
) -> Tuple[Tensor, Tensor]:
    """Compute mean and standard deviation of a dataset.

    Args:
        dataloader: torch dataloader.
        batch_key: key in the `batch` dictionary where the relevant data is 
            stored, and which is used to to compute the statistics.
        data_type: type of data over which to compute statistics.
    """

    num_samples = 0
    mean = 0.
    std = 0.

    for batch in dataloader:
        
        # Access relevat data
        data = batch[batch_key]
        
        # Update total number of samples
        num_samples += data.size(0)
        
        if data_type == "images":

            # Rearrange tensor to be the shape of [B, C, W * H]
            data = data.view(data.size(0), data.size(1), -1)
             # Compute mean and std per channel
            mean += data.mean(2).sum(0) 
            std += data.std(2).sum(0)
            
        else:
            raise NotImplementedError

    mean /= num_samples
    std /= num_samples

    print(mean)
    print(std)

    return (mean, std)