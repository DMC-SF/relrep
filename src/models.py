import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST


def relative_projection(x, anchors):
    x = F.normalize(x, dim=1)
    anchors = F.normalize(anchors, dim=1)
    return torch.einsum("im,jm -> ij", x, anchors)


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size=32, anchors=None, use_relative_space=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, hidden_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        self.use_relative_space = use_relative_space
        if anchors is not None:
            self.register_buffer("anchors", anchors)

    def forward(self, x):
        encoded = self.encoder(x)
        if self.use_relative_space:
            encoded = relative_projection(encoded, self.anchors)
        decoded = self.decoder(encoded)
        return decoded


class MNISTModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        y = self.net(x)
        loss = F.mse_loss(y, x)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        y = self.net(x)
        loss = F.mse_loss(y, x)
        self.log("val/loss", loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.hparams.lr)
    
    def train_dataloader(self):
        dataset = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        dataset = MNIST("data", train=False, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)