import os 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

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
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        for i, (img, label) in enumerate(train_loader):
            if i == n_anchors:
                break
            torch.save(img, f"data/anchors/anchor_{i}.pt")

def load_anchors():
    """Load all the anchors, of the form anchor_n, from the anchors folder. 
    The n number are note sequential, so we use a dictionary to store the anchors.
    return a Tensor of dimension (n_anchors, 1, 28, 28)."""
    anchors = []
    for img in os.listdir("data/anchors"):
        print(img)
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

# def select_random_anchors(
#     train_dataset: Dataset,
#     n_anchors: int = 10,
#     data_dir: str = "data",
#     anchors_dir: str = "anchors",
# ):
#     """Save n_anchors images from the train_dataset in the save_path folder.
#     if n_anchors is lesser or equal to 10, we choose 10 img with different labels. Otherwise, we choose n_anchors random image.
#     The images are saved as anchor_{i}.png where i is the index of the image in the training dataset.
#     The images are saved with the transform=transforms.ToTensor()"""

#     if not os.path.exists(f"{data_dir}/{anchors_dir}"):
#         os.makedirs(f"{data_dir}/{anchors_dir}")

#     if n_anchors <= 10:
#         # select 10 images with different labels
#         labels = []
#         for i, (img, label) in enumerate(train_dataset):
#             if label not in labels:
#                 labels.append(label)
#                 torchvision.utils.save_image(img, f"{data_dir}/{anchors_dir}/anchor_{i}.png")
#             if len(labels) == n_anchors:
#                 break
#     else:
#         # select n_anchors random images
#         train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#         for i, (img, label) in enumerate(train_loader):
#             if i == n_anchors:
#                 break
#             torchvision.utils.save_image(img, f"{data_dir}/{anchors_dir}/anchor_{i}.png")


# def _load_anchors(data_dir: str = "data", anchors_dir: str = "anchors"):
#     """Load all the anchors from the anchors_dir folder. return a Tensor of dimension (n_anchors, 1, 28, 28). 
#     Apply the same transformations as the training dataset: transform=transforms.ToTensor().
#     avoid: TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>"""
#     anchors = []
#     normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
#     for img in os.listdir(f"{data_dir}/{anchors_dir}"):
#         img = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Grayscale(),
#                 normalize,
#             ])(plt.imread(f"{data_dir}/{anchors_dir}/{img}"))
#         anchors.append(img)
#     anchors = torch.stack(anchors)
#     return anchors