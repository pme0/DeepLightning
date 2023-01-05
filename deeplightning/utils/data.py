from torch.utils.data import DataLoader


def compute_dataset_mean_and_stdev(dataloader: DataLoader, images_index: int):
    """Compute mean and standard deviation of image dataset.

    Parameters
    ----------
    dataloader : torch dataloader.
    
    images_index : index of tuple corresponding to images tensor.

    """

    nimages = 0
    mean = 0.
    std = 0.

    for batch in dataloader:
        images = batch[images_index]
        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total number of images
        nimages += images.size(0)
        # Compute mean and std here
        mean += images.mean(2).sum(0) 
        std += images.std(2).sum(0)

    mean /= nimages
    std /= nimages

    print(mean)
    print(std)