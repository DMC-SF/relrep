import json
import os
from collections import defaultdict
import warnings

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
from utils import load_anchors, select_random_anchors, strip_and_load
from models import AutoEncoder, MNISTModule



def train(seed=42, use_relative_space=True, data_dir="data", anchors_dir="anchors"):
    """Train the ConvAutoencoder model and saves weights in the weights folder."""
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*Tensor Cores. To properly*")

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

    trainer = pl.Trainer(
        max_epochs=50, 
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
    )

    trainer.fit(module)

    if not os.path.exists("weights"):
        os.makedirs("weights")
    
    save_path = f"weights/conv_autoencoder_seed={seed}_relative_space={use_relative_space}.pt"
    torch.save(module.state_dict(), save_path)


def compare_models(
    encoder_weights_path: str, 
    decoder_weights_path: str, 
    use_relative_space: bool = True,
    n_images: int = 10
):
    """
    Compose an autoencoder model from two differently trained encoder and decoder and plot
    original image vs reconstruction for n_images samples in the validation dataset.
    """

    model = AutoEncoder(hidden_size=10)
    strip_and_load(model, encoder_weights_path)
    strip_and_load(model, decoder_weights_path)

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
    plt.savefig(f"images/experiment_relative_space={use_relative_space}.png")


def experiment():
    """
    We first train two models with different seeds and no relative space.
    Then, we do the same with relative space.
    Finally, for both cases we exchange encoder and decoder weights and compare the
    reconstruction.
    """
    # Training without relative space
    # train(seed=42, use_relative_space=False)
    # train(seed=24, use_relative_space=False)

    # Training with relative space
    # train(seed=42, use_relative_space=True)
    # train(seed=24, use_relative_space=True)

    # Compare models
    compare_models(
        encoder_weights_path="weights/conv_autoencoder_seed=42_relative_space=False.pt",
        decoder_weights_path="weights/conv_autoencoder_seed=24_relative_space=False.pt",
        use_relative_space=False,
    )

    compare_models(
        encoder_weights_path="weights/conv_autoencoder_seed=42_relative_space=True.pt",
        decoder_weights_path="weights/conv_autoencoder_seed=24_relative_space=True.pt",
        use_relative_space=True,
    )


if __name__ == "__main__":
    experiment()
