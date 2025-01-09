import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import time

# Hyperparameters
batch_size = 256
learning_rate = 0.005
epochs = 30
num_classes = 10
hidden_neurons = 500 
beta_val=1.0

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./DB", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./DB", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

start_time=time.time()

# Flatten the images for PCA
train_data = train_dataset.data.reshape(len(train_dataset), -1)
test_data = test_dataset.data.reshape(len(test_dataset), -1)

# Apply Incremental PCA with 100 components
incremental_pca = IncrementalPCA(n_components=100)
batch_size_pca = 1000

# Fit PCA in chunks
for i in range(0, len(train_data), batch_size_pca):
    incremental_pca.partial_fit(train_data[i:i + batch_size_pca])
train_data_pca = incremental_pca.transform(train_data)
test_data_pca = incremental_pca.transform(test_data)

# Normalize PCA data
train_data_pca = (train_data_pca - train_data_pca.mean(axis=0)) / train_data_pca.std(axis=0)
test_data_pca = (test_data_pca - test_data_pca.mean(axis=0)) / test_data_pca.std(axis=0)

# Use MiniBatchKMeans to initialize RBF centers
mini_batch_kmeans = MiniBatchKMeans(n_clusters=hidden_neurons, batch_size=batch_size_pca, random_state=42)
mini_batch_kmeans.fit(train_data_pca)
centers_init = torch.tensor(mini_batch_kmeans.cluster_centers_, dtype=torch.float32)

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
        self.beta = nn.Parameter(torch.full((1,), beta_val))  # Initialize beta to a sensible value
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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay

# Training Loop with Mini-Batch Processing
losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / (len(train_data) / batch_size)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

total_time=time.time() - start_time
minutes= total_time//60
seconds= total_time - minutes*60
print(f"Training complete in {int(minutes)} minutes, {seconds:.2f} seconds")

# Plot Training Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig("training_loss.png")  # Save the plot to a file

# Testing Loop
print("Starting evaluation...")
model.eval()
with torch.no_grad():
    test_outputs = model(test_data)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == test_labels).float().mean()

print(f"Accuracy with RBF: {accuracy * 100:.2f}%")
