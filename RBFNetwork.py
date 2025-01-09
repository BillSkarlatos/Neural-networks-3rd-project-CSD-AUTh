import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 10
num_classes = 10
hidden_neurons = 100

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./DB", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./DB", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Flatten the images for PCA
train_data = train_dataset.data.reshape(len(train_dataset), -1)
test_data = test_dataset.data.reshape(len(test_dataset), -1)

# Apply PCA with 95% variance retained
pca = PCA(n_components=0.95)
train_data_pca = pca.fit_transform(train_data)
test_data_pca = pca.transform(test_data)

# Normalize PCA data
train_data_pca = (train_data_pca - train_data_pca.mean(axis=0)) / train_data_pca.std(axis=0)
test_data_pca = (test_data_pca - test_data_pca.mean(axis=0)) / test_data_pca.std(axis=0)

# Use K-Means to initialize RBF centers
kmeans = KMeans(n_clusters=hidden_neurons, random_state=42)
kmeans.fit(train_data_pca)
centers_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

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

train_data, train_labels = prepare_data(train_data_pca, train_dataset.targets)
test_data, test_labels = prepare_data(test_data_pca, test_dataset.targets)

# Model, Loss, Optimizer
model = RBFNetwork(input_dim=train_data.shape[1], hidden_neurons=hidden_neurons, num_classes=num_classes, centers_init=centers_init)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay | Recommended by ChatGPT

# Training Loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Testing Loop
print("Starting evaluation...")
model.eval()
with torch.no_grad():
    test_outputs = model(test_data)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == test_labels).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
