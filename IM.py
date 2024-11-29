import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import struct
import os
import random
import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm  

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
n_epochs = 50 


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

full_train_dataset = torchvision.datasets.MNIST(
    './data', train=True, download=True,
    transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
)

# Group indices by original class labels
class_indices = {i: [] for i in range(10)}
for idx, (image, label) in enumerate(full_train_dataset):
    class_indices[label].append(idx)

# Downsample class '1'
original_class_1_count = len(class_indices[1])
downsample_size = int(original_class_1_count * 0.1)  # 10% of the original class '1' samples

downsampled_class_1_indices = np.random.choice(class_indices[1], downsample_size, replace=False)
remaining_indices = [idx for label, indices in class_indices.items() if label != 1 for idx in indices]

final_indices = list(downsampled_class_1_indices) + remaining_indices
np.random.shuffle(final_indices)
downsampled_train_dataset = Subset(full_train_dataset, final_indices)

train_size = int(len(downsampled_train_dataset) * 0.8)
val_size = len(downsampled_train_dataset) - train_size
train_indices, val_indices = random_split(range(len(downsampled_train_dataset)), [train_size, val_size])

downsampled_original_labels = [full_train_dataset[idx][1] for idx in final_indices]
train_original_labels = [downsampled_original_labels[idx] for idx in train_indices]
val_original_labels = [downsampled_original_labels[idx] for idx in val_indices]

# Binary mapping function 
def binary_label_mapping(dataset_indices):
    dataset_with_binary_labels = []
    for idx in dataset_indices:
        img, label = downsampled_train_dataset[idx]
        binary_label = 0 if label % 2 == 0 else 1
        dataset_with_binary_labels.append((img, binary_label))
    return dataset_with_binary_labels

train_dataset = binary_label_mapping(train_indices)
val_dataset = binary_label_mapping(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False)

test_dataset = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
)

# Also mapping the test set of course
test_dataset = binary_label_mapping(range(len(test_dataset)))
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)

print(f"Training Dataset Size: {train_size} samples")
print(f"Validation Dataset Size: {val_size} samples")
print(f"Test Dataset Size: {test_size} samples")


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


network = Net()
network.load_state_dict(torch.load('model2.pth')) 
network.eval()

hessian_diag_full = np.load('hessian_diag_full_HS_binary2.npy')
print(f"Hessian approximation (full dataset): {hessian_diag_full}")
inverse_hessian_diag = 1 / hessian_diag_full
print("Inverse Diagonal Hessian:", inverse_hessian_diag)

print("Processing data...")

subset_size = 1000  

subset_indices = []
subset_data = []
subset_labels = []

for batch_idx, (z_train_batch, target_train_batch) in enumerate(tqdm(train_loader, desc="Data Loader")):
    for train_index in range(len(z_train_batch)):
        subset_indices.append(batch_idx * len(z_train_batch) + train_index)
        subset_data.append(z_train_batch[train_index])
        subset_labels.append(target_train_batch[train_index].item())

subset_data = torch.stack(subset_data)
subset_labels = torch.tensor(subset_labels)

# Gradients computation
print("Computing gradients...")
network.eval()
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
        influence = -torch.dot(gradients[i], gradients[j] * inverse_hessian_diag).item()
        influence_matrix[i, j] = influence
        influence_matrix[j, i] = influence  # Symmetry

print("Influence matrix computation completed.")

output_dir = "/work3/s206182/projects/detecting_causes_of_gender_bias_chest_xrays"
np.save(f"{output_dir}/influence_matrix.npy", influence_matrix.numpy())
np.save(f"{output_dir}/sample_indices.npy", subset_indices)
