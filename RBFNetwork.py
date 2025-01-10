import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Radial Basis Function (RBF) Layer
def rbf_kernel(x, centers, beta):
    """Compute the RBF kernel."""
    x = x.unsqueeze(1)
    return torch.exp(-beta * torch.sum((x - centers) ** 2, dim=2))

class RBFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_neurons, num_classes, centers_init):
        super(RBFNetwork, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.centers = nn.Parameter(centers_init)
        self.beta = nn.Parameter(torch.ones(1))
        self.fc = nn.Linear(hidden_neurons, num_classes)

    def forward(self, x):
        rbf_output = rbf_kernel(x, self.centers, self.beta)
        return self.fc(rbf_output)

# Prepare Data for PyTorch
def prepare_data(data, labels):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor

def compute_adaptive_beta(data, subset_size=2000):
    """Compute beta for adaptive learning using a subset of data."""
    sampled_data = data[np.random.choice(len(data), subset_size, replace=False)]
    pairwise_distances = torch.cdist(torch.tensor(sampled_data), torch.tensor(sampled_data))
    mean_distance = pairwise_distances.mean().item()
    beta = 1.0 / (2 * (mean_distance ** 2))
    return beta

def RBF(training_mode = "kmeans", hidden_neurons = 200, learning_rate = 0.002, batch_size = 256, epochs = 20): 
    # Hyperparameters
    num_classes = 10
    chunk_size = 1000  # Smaller chunk size for Incremental PCA to save memory

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./DB", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./DB", train=False, download=True, transform=transform)

    # Flatten the images for Incremental PCA
    train_data = train_dataset.data.reshape(len(train_dataset), -1).astype(np.float32)
    test_data = test_dataset.data.reshape(len(test_dataset), -1).astype(np.float32)

    # Determine the number of components for 95% variance using standard PCA on a small subset
    print("Applying PCA . . .")
    start_time=time.time()
    subset = train_data[:1000]  # Use a small subset to determine n_components
    pca_temp = PCA(n_components=0.95)
    pca_temp.fit(subset)
    n_components = pca_temp.n_components_

    # Apply Incremental PCA with the determined number of components
    ipca = IncrementalPCA(n_components=n_components)
    for i in range(0, len(train_data), chunk_size):
        ipca.partial_fit(train_data[i:i + chunk_size])

    train_data_pca = np.vstack([ipca.transform(train_data[i:i + chunk_size]) for i in range(0, len(train_data), chunk_size)])
    test_data_pca = np.vstack([ipca.transform(test_data[i:i + chunk_size]) for i in range(0, len(test_data), chunk_size)])

    # Normalize PCA data
    train_data_pca = (train_data_pca - train_data_pca.mean(axis=0)) / train_data_pca.std(axis=0)
    test_data_pca = (test_data_pca - test_data_pca.mean(axis=0)) / test_data_pca.std(axis=0)

    # Initialize RBF centers based on training mode
    if training_mode == "kmeans":
        kmeans = KMeans(n_clusters=hidden_neurons, random_state=42)
        kmeans.fit(train_data_pca)
        centers_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    elif training_mode == "random":
        random_indices = np.random.choice(len(train_data_pca), hidden_neurons, replace=False)
        centers_init = torch.tensor(train_data_pca[random_indices], dtype=torch.float32)
    elif training_mode == "adaptive":
        beta = compute_adaptive_beta(train_data_pca, subset_size=2000)
        centers_init = torch.tensor(train_data_pca[:hidden_neurons], dtype=torch.float32)
    else:
        raise ValueError("Invalid training_mode. Choose from 'kmeans', 'random', or 'adaptive'.")

    train_data, train_labels = prepare_data(train_data_pca, train_dataset.targets)
    test_data, test_labels = prepare_data(test_data_pca, test_dataset.targets)

    # Create DataLoader for batch processing
    train_dataset_pca = TensorDataset(train_data, train_labels)
    test_dataset_pca = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset_pca, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_pca, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = RBFNetwork(input_dim=train_data.shape[1], hidden_neurons=hidden_neurons, num_classes=num_classes, centers_init=centers_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay

    # Training Loop
    print("Starting Training . . .")
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    total_time=time.time() - start_time
    minutes= total_time//60
    seconds= total_time - minutes*60
    print(f"Training complete in {int(minutes)} minutes, {seconds:.2f} seconds")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.savefig('training_loss_curve.png')

    # Testing Loop
    print("Starting evaluation...")
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            test_outputs = model(batch_data)
            _, predicted = torch.max(test_outputs, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Accuracy of the Radial Basis Function Neural Network ({training_mode} training mode): {accuracy*100:.2f}%")
