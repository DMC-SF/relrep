import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Functions for selecting and saving fixed points indices
def select_balanced_indices(dataset, num_points_per_class=1):
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_indices.items():
        # np.random.shuffle(indices)
        selected_indices.extend(indices[:num_points_per_class])

    return selected_indices

def save_indices(indices, filename='selected_indices.json'):
    with open(filename, 'w') as f:
        json.dump(indices, f)

def load_indices(filename='selected_indices.json'):
    with open(filename, 'r') as f:
        indices = json.load(f)
    return indices

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

mnist_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(mnist_data))
val_size = len(mnist_data) - train_size
train_dataset, val_dataset = random_split(mnist_data, [train_size, val_size])

# Select and save fixed_points_indices
num_points_per_class = 1
fixed_points_indices = select_balanced_indices(train_dataset, num_points_per_class=num_points_per_class)
save_indices(fixed_points_indices, 'fixed_points_indices.json')

# Load fixed_points_indices
fixed_points_indices = load_indices('fixed_points_indices.json')



class Encoder(nn.Module):
    def __init__(self, hidden_size=32):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, hidden_size)
        )
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, hidden_size=32):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


class ConvAutoencoder(pl.LightningModule):
    def __init__(self, hidden_size=32):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size)
        self.fixed_points_indices = fixed_points_indices

    def forward(self, x):
        x = self.encoder(x)
        fixed_points = self.encode_fixed_points()
        x = self.cosine_similarity_layer(x, fixed_points)
        x = self.decoder(x)
        return x

    def cosine_similarity_layer(self, x, fixed_points):
        similarities = torch.mm(x, fixed_points.t())  # Compute cosine similarity
        return similarities

    def encode_fixed_points(self):
        with torch.no_grad():
            fixed_points = []

            for idx in self.fixed_points_indices:
                img, _ = train_dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                encoded = self.encoder(img)
                fixed_points.append(encoded.cpu())

            fixed_points = torch.cat(fixed_points, dim=0)
            fixed_points.requires_grad = False
            return fixed_points
        

# Training with Lightning
model = ConvAutoencoder()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, DataLoader(train_dataset, batch_size=32), DataLoader(val_dataset, batch_size=32))
