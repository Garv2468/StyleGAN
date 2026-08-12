"""Dataset download and DataLoader construction."""

import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .Model import CHANNELS


def download_dataset(dataset_name="manavkumarjalan93/anime-face-dataset"):
    """Download the dataset via kagglehub and return its local path."""
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to: {path}")
    return path


def get_dataloader(root, image_size=None, batch_size=32):
    if image_size is None:
        image_size = 4 * (2 ** (len(CHANNELS) - 1))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)