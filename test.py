import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

# Hyperparameters
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
n_epochs = 50
random_seed = 42

# Set random seeds
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Load MNIST dataset
full_train_dataset = torchvision.datasets.MNIST(
    './data', train=True, download=True,
    transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
)

# Use a subset of 10,000 samples
subset_size = 10000
subset_indices = np.random.choice(len(full_train_dataset), subset_size, replace=False)
subset_train_dataset = Subset(full_train_dataset, subset_indices)

# Create DataLoader for the subset
train_loader = DataLoader(subset_train_dataset, batch_size=batch_size_train, shuffle=True)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(32, 10, kernel_size=3, padding=1)  

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

# Load the pre-trained network
network = Net()
network.load_state_dict(torch.load('model2.pth')) 
network.eval()

# Load the precomputed Hessian diagonal
hessian_diag_full = np.load('hessian_diag_full_HS_binary2.npy')
print(f"Hessian approximation (full dataset): {hessian_diag_full}")
inverse_hessian_diag = 1 / hessian_diag_full
print("Inverse Diagonal Hessian:", inverse_hessian_diag)

# Collect subset data and labels
subset_data = []
subset_labels = []
for batch_idx, (z_train_batch, target_train_batch) in enumerate(tqdm(train_loader, desc="Loading Data")):
    subset_data.append(z_train_batch)
    subset_labels.append(target_train_batch)

subset_data = torch.cat(subset_data)
subset_labels = torch.cat(subset_labels)

# Gradients computation
print("Computing gradients...")
gradients = []

for idx in tqdm(range(len(subset_data)), desc="Gradient Computation"):
    z_train = subset_data[idx].unsqueeze(0)
    target_train = torch.tensor([subset_labels[idx]])

    network.zero_grad()
    output_train = network(z_train)
    loss_train = F.nll_loss(output_train, target_train)
    loss_train.backward()
    grad_train = torch.cat([param.grad.view(-1) for param in network.parameters()])
    gradients.append(grad_train)

gradients = torch.stack(gradients)

# Influence matrix computation
print("Computing influence matrix...")
dataset_size = len(subset_data)
influence_matrix = torch.zeros((dataset_size, dataset_size))

for i in tqdm(range(dataset_size), desc="Influence Matrix Row"):
    for j in range(i, dataset_size):
        influence = -torch.dot(gradients[i], gradients[j] * torch.tensor(inverse_hessian_diag)).item()
        influence_matrix[i, j] = influence
        influence_matrix[j, i] = influence  # Symmetry

print("Influence matrix computation completed.")

# Save the results
output_dir = "/work3/s206182/projects/detecting_causes_of_gender_bias_chest_xrays"
np.save(f"{output_dir}/subset_influence_matrix.npy", influence_matrix.numpy())
np.save(f"{output_dir}/subset_sample_indices.npy", subset_indices)

print("Results saved.")
