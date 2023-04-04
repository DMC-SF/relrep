import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from utils.utils import load_anchors, strip_and_load
from models.autoencoder import AutoEncoder as AE
from models.variational import VariationalAutoEncoder as VAE
import hydra

os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base="1.3", config_path="../configs", config_name="plot.yaml")
def experiment(cfg):
    """
    We first train two models with different seeds and no relative space.
    Then, we do the same with relative space.
    Finally, for both cases we exchange encoder and decoder weights and compare the
    reconstruction.
    """

    enc_relative_space = cfg.enc_relative_space
    enc_seed = cfg.enc_seed
    dec_relative_space = cfg.dec_relative_space
    dec_seed = cfg.dec_seed    

    compare_models(
        encoder_weights_path=f"weights/enc_seed={enc_seed}_rs={enc_relative_space}.pt",
        decoder_weights_path=f"weights/dec_seed={dec_seed}_rs={dec_relative_space}.pt",
        tag=cfg.tag,
        use_relative_space=cfg.use_relative_space,
        variational=cfg.variational,
        num_anchor=cfg.num_anchor,
    )


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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    experiment()
