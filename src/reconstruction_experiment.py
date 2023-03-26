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
from utils import load_anchors, select_random_anchors
from models import AutoEncoder, MNISTModule


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
    BATCH_SIZE = 128

    if not os.path.exists(f"{data_dir}/{anchors_dir}"):
        os.makedirs(f"{data_dir}/{anchors_dir}")
        select_random_anchors(
            MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor()),
            n_anchors=N_ANCHORS,
            data_dir=data_dir, 
            anchors_dir=anchors_dir
        )

    anchors = load_anchors(data_dir=data_dir, anchors_dir=anchors_dir)

    module = MNISTModule(
        net = AutoEncoder(hidden_size=N_ANCHORS, anchors=anchors),
        batch_size=BATCH_SIZE,
    )

    trainer = pl.Trainer(max_epochs=50, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(module)

    if not os.path.exists("weights"):
        os.makedirs("weights")

    torch.save(module.state_dict(), f"weights/conv_autoencoder_seed={seed}.pt")


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

    # Save the plot
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig("images/stitched_autoencoder.png")


if __name__ == "__main__":
    #train(seed=42, data_dir="data", anchors_dir="anchors")
    #train(seed=43, data_dir="data", anchors_dir="anchors")

    compare_stitched_models(
        encoder_weights_path="weights/conv_autoencoder_seed=42.pt",
        decoder_weights_path="weights/conv_autoencoder_seed=43.pt",
    )

