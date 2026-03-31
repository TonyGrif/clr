from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_loaders(
    batch_size: int,
    seed: int,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-10 train, validation, and test DataLoaders.

    Splits the 50,000 training images into 45,000 train and 5,000 validation.
    The original 10,000 test images are used as the test set.

    Args:
        batch_size: Number of samples per batch.
        seed: Random seed for the train/val split.
        data_dir: Directory to download/cache the dataset.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    train_full = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_set, val_set = random_split(
        train_full,
        [45000, 5000],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
