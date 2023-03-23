from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
from src.module import ConvAutoencoder
from src.datamodules import MNISTDataModule

ROOT_DIR = Path('.')
DATA_DIR = ROOT_DIR / 'data'
WEIGHTS_DIR = ROOT_DIR / 'weights'
LOG_DIR = ROOT_DIR / 'logs'


def train():
    batch_size = 128
    num_epochs = 2

    mnist_dm = MNISTDataModule(data_dir=DATA_DIR, batch_size=batch_size)
    model = ConvAutoencoder()
    logger = TensorBoardLogger(save_dir=LOG_DIR)

    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        gpus=int(torch.cuda.is_available()),
        
    )
    trainer.fit(model, mnist_dm)

    # Save the model weights
    torch.save(model.state_dict(), WEIGHTS_DIR / 'conv_autoencoder.pth')


def plot_latent_space(model, val_dataset):
    model.eval()
    with torch.no_grad():
        encoded_points = []
        labels = []

        for img, label in val_dataset:
            img = img.unsqueeze(0).to(model.device)
            encoded = model.encoder(img)
            encoded_points.append(encoded.cpu().numpy())
            labels.append(label)

        encoded_points = np.vstack(encoded_points)
        labels = np.array(labels)

        fig, ax = plt.subplots()
        scatter = ax.scatter(encoded_points[:, 0], encoded_points[:, 1], c=labels, cmap='tab10', alpha=0.5)
        legend1 = ax.legend(*scatter.legend_elements(), title="Class", loc="upper left", bbox_to_anchor=(1, 1))
        ax.add_artist(legend1)

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Latent Space of Validation Dataset')
        plt.show()

