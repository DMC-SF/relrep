import os 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt


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