import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))

x_train_flat = x_train.view(x_train.size(0), -1).numpy()
x_test_flat = x_test.view(x_test.size(0), -1).numpy()

# Apply PCA to reduce dimensionality
pca = PCA(0.9)  # Keep 90% of variance
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

x_train_pca = torch.tensor(x_train_pca, dtype=torch.float32)
x_test_pca = torch.tensor(x_test_pca, dtype=torch.float32)
y_train_onehot = torch.nn.functional.one_hot(y_train, num_classes=10).float()

def hebbian_learning(inputs, targets, learning_rate=0.01, n_epochs=10):
    """
    Hebbian learning implementation using PyTorch tensors.

    Args:
        inputs (Tensor): Input features (N_samples x N_features).
        targets (Tensor): Target outputs (N_samples x N_classes).
        learning_rate (float): Learning rate.
        n_epochs (int): Number of training epochs.

    Returns:
        Tensor: Learned weight matrix.
    """
    n_features = inputs.size(1)
    n_classes = targets.size(1)

    # Initialize weights randomly
    weights = torch.randn(n_features, n_classes) * 0.01

    for epoch in range(n_epochs):
        for i in range(inputs.size(0)):
            x = inputs[i].unsqueeze(1)  # Column vector
            y = targets[i].unsqueeze(1)  # Column vector
            weights += learning_rate * x @ y.T  # Update weights

    return weights

# Train Hebbian network
weights = hebbian_learning(x_train_pca, y_train_onehot, learning_rate=0.01, n_epochs=10)

# Testing function
def predict(inputs, weights):
    """Predict class labels using learned weights."""
    outputs = inputs @ weights
    return torch.argmax(outputs, dim=1)

# Make predictions
y_pred = predict(x_test_pca, weights)

# Calculate accuracy
accuracy = (y_pred == y_test).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize some results
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].permute(1, 2, 0) * 0.5 + 0.5)  # Denormalize for visualization
    plt.title(f"Pred: {y_pred[i].item()}, True: {y_test[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()