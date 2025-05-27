"""
    Provides train, validation, and test data loaders for CIFAR-10.
"""

import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=128, val_size=1024):
    """
        Load CIFAR-10 dataset and return data loaders for training, validation, and testing.

        Args:
            batch_size (int): Batch size for data loaders. Default is 128.
            val_size (int): Number of samples to reserve for validation. Default is 1024.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
    """

    # define normalization and conversion transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load full CIFAR-10 training and test sets
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # create index masks for validation and training splits
    indices = list(range(len(dataset)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader