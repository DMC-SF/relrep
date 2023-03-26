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


class Encoder(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, hidden_size),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class ConvAutoencoder(pl.LightningModule):
    def __init__(
        self, hidden_size: int = 32, anchors: torch.Tensor = None, use_relative_space: bool = False
    ):
        super().__init__()
        self.encoder = Encoder(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size)
        self.anchors = anchors
        self.use_relative_space = use_relative_space

    def forward(self, x):
        x = self.encoder(x)
        if self.use_relative_space:
            x = self.relative_projection(x, self.anchors)
        x = self.decoder(x)
        return x

    @staticmethod
    def relative_projection(x, anchors):
        x = F.normalize(x, dim=1)
        anchors = F.normalize(anchors, dim=1)
        return torch.einsum("im,jm -> ij", x, anchors)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def select_random_anchors(
    train_dataset: Dataset,
    n_anchors: int = 10,
    data_dir: str = "data",
    anchors_dir: str = "anchors",
):
    """Save n_anchors images from the training dataset in the save_path folder."""
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, (img, label) in enumerate(train_loader):
        if i == n_anchors:
            break
        torchvision.utils.save_image(img, f"{data_dir}/{anchors_dir}/anchor_{i}.png")


def load_anchors(data_dir: str = "data", anchors_dir: str = "anchors"):
    """Load the anchors from the anchors_dir folder."""
    anchors = []
    for anchor in os.listdir(f"{data_dir}/{anchors_dir}"):
        img = torchvision.io.read_image(f"{data_dir}/{anchors_dir}/{anchor}")
        anchors.append(img)
    return torch.stack(anchors)


# We can now define the training loop
# We first load the training and validation datasets
# We then select the anchors and save them in the anchors folder
# We then create the model and pass the anchors to the model
# We then train the model
# We then plot the latent space of the validation dataset


def train(seed=42, data_dir="data", anchors_dir="anchors"):
    """Train the ConvAutoencoder model and saves weights in the weights folder."""
    pl.seed_everything(seed)
    N_ANCHORS = 10
    train_dataset = MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    val_dataset = MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    if not os.path.exists(f"{data_dir}/{anchors_dir}"):
        os.makedirs(f"{data_dir}/{anchors_dir}")
        select_random_anchors(
            train_dataset, n_anchors=N_ANCHORS, data_dir=data_dir, anchors_dir=anchors_dir
        )

    anchors = load_anchors(data_dir=data_dir, anchors_dir=anchors_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = ConvAutoencoder(hidden_size=N_ANCHORS, anchors=anchors)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_loader, val_loader)

    if not os.path.exists("weights"):
        os.makedirs("weights")

    torch.save(model.state_dict(), f"weights/conv_autoencoder_seed={seed}.pt")


def load_weights(model, weights_path):
    """Load the weights from the weights_path file into the model."""
    model.load_state_dict(torch.load(weights_path))
    return model


def compare_stitched_models(
    encoder_weights_path: str, decoder_weights_path: str, n_images: int = 10
):
    """Compose an autoencoder model from two differently trained encoder and decoder and plot
    original image vs reconstruction for n_images samples in the validation dataset."""

    model = ConvAutoencoder(hidden_size=10)
    model.load_state_dict(torch.load(encoder_weights_path))
    model.load_state_dict(torch.load(decoder_weights_path))

    val_dataset = MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    fig, axs = plt.subplots(2, n_images, figsize=(20, 4))
    for i, (img, label) in enumerate(val_loader):
        if i == n_images:
            break
        img_hat = model(img)
        axs[0, i].imshow(img[0, 0], cmap="gray")
        axs[1, i].imshow(img_hat[0, 0].detach(), cmap="gray")

    plt.show()


if __name__ == "__main__":
    # train(seed=42, data_dir="data", anchors_dir="anchors")
    # train(seed=43, data_dir="data", anchors_dir="anchors")

    compare_stitched_models(
        encoder_weights_path="weights/conv_autoencoder_seed=42.pt",
        decoder_weights_path="weights/conv_autoencoder_seed=43.pt",
    )
