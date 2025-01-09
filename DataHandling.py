import torch
from torchvision import datasets, transforms
import numpy as np



def loadDatabase():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root="./DB", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./DB", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader

def load_data(data_loaders, limit=0):
    """
    Extracts data and labels from PyTorch data loaders and converts them to NumPy arrays.

    Args:
        data_loaders (tuple): A tuple containing training and testing data loaders.
        limit (int): Maximum number of samples to load (0 for no limit).

    Returns:
        tuple: Input data, input labels, test data, test labels as NumPy arrays.
    """
    train_loader, test_loader = data_loaders
    
    # Extract training data
    train_data = []
    train_labels = []
    for data, labels in train_loader:
        train_data.append(data.view(data.size(0), -1).numpy())  # Flatten images
        train_labels.append(labels.numpy())
        if limit > 0 and len(train_data) * train_loader.batch_size >= limit:
            break
    
    # Extract test data
    test_data = []
    test_labels = []
    for data, labels in test_loader:
        test_data.append(data.view(data.size(0), -1).numpy())  # Flatten images
        test_labels.append(labels.numpy())
        if limit > 0 and len(test_data) * test_loader.batch_size >= limit:
            break

    # Concatenate batches into arrays
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    return train_data, train_labels, test_data, test_labels
