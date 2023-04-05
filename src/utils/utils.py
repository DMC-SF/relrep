import os 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from models.autoencoder import AutoEncoder as AE
from models.variational import VariationalAutoEncoder as VAE

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def compare_models(
    encoder_weights_path: str, 
    decoder_weights_path: str, 
    tag: str,
    use_relative_space: bool = True,
    variational: bool = False,
    num_anchor: int = 10,
    n_images: int = 10,
):
    """
    Compose an autoencoder model from two differently trained encoder and decoder and plot
    original image vs reconstruction for n_images samples in the validation dataset.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    anchors = load_anchors()
    if variational:
        model = VAE(anchors=anchors, hidden_size=num_anchor, use_relative_space=use_relative_space).to(device)
    else:
        model = AE(anchors=anchors, hidden_size=num_anchor, use_relative_space=use_relative_space).to(device)
    model = strip_and_load(model, encoder_weights_path, decoder_weights_path)

    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    val_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    fig, axs = plt.subplots(2, n_images, figsize=(20, 4))
    for i, (img, _) in enumerate(val_loader):
        if i == n_images:
            break
        if variational:
            img_hat = model(img.to(device))[0].cpu()
        else:
            img_hat = model(img.to(device)).cpu()
        axs[0, i].imshow(img[0, 0], cmap="gray")
        axs[1, i].imshow(img_hat[0, 0].detach(), cmap="gray")

    # Save the plot
    if not os.path.exists("images"):
        os.makedirs("images")
    if variational:
        plt.savefig(f"images/var_rs={use_relative_space}_{tag}.png")
    else:
        plt.savefig(f"images/rs={use_relative_space}_{tag}.png")
    
    print(f'Images saved in \"images/rs={use_relative_space}_{tag}.png\"')

def strip_and_load(model, weights_enc_path, weights_dec_path):
    """
    Loads weights from weights_path into model. 
    return model with weights loaded with a specific encoder and decoder.
    """
    weights_enc = torch.load(weights_enc_path)
    weights_enc = {k.replace("net.", ""): v for k, v in weights_enc.items()}
    model.encoder.load_state_dict(weights_enc,  strict=False)
    weights_dec = torch.load(weights_dec_path)
    weights_dec = {k.replace("net.", ""): v for k, v in weights_dec.items()}
    model.decoder.load_state_dict(weights_dec,  strict=False)
    return model

def save_anchors(
        train_dataset: Dataset,
        n_anchors: int = 10,
    ):
    """
    Save n_anchors as Tensors (torch.save(img)) in the anchors folder.
    if n_anchors is lesser or equal to 10, we choose 10 img with different labels. Otherwise, we choose n_anchors random image.
    """
    if not os.path.exists("data/anchors"):
        os.makedirs("data/anchors")

    if n_anchors <= 10:
        # select 10 images with different labels
        labels = []
        for i, (img, label) in enumerate(train_dataset):
            if label not in labels:
                labels.append(label)
                torch.save(img, f"data/anchors/anchor_{i}.pt")
            if len(labels) == n_anchors:
                break
    else:
        # select n_anchors random images
        for i, (img, label) in enumerate(train_dataset):
            if i == n_anchors:
                break
            torch.save(img, f"data/anchors/anchor_{i}.pt")

def load_anchors():
    """Load all the anchors, of the form anchor_n, from the anchors folder. 
    The n number are note sequential, so we use a dictionary to store the anchors.
    return a Tensor of dimension (n_anchors, 1, 28, 28)."""
    anchors = []
    for img in os.listdir("data/anchors"):
        anchors.append(torch.load(f"data/anchors/{img}"))
    anchors = torch.stack(anchors)
    return anchors

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