import torch
import numpy as np
import matplotlib.pyplot as plt


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