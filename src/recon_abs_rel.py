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
from utils.utils import compare_models

os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base="1.3", config_path="../configs", config_name="plot.yaml")
def experiment(cfg):
    """
    We first train two models with different seeds and no relative space.
    Then, we do the same with relative space.
    Finally, for both cases we exchange encoder and decoder weights and compare the
    reconstruction.
    """

    relative_space = cfg.relative_space

    compare_models(
        encoder_weights_path=f"weights/enc_seed={cfg.seed_1}_rs={relative_space}.pt",
        decoder_weights_path=f"weights/dec_seed={cfg.seed_1}_rs={relative_space}.pt",
        tag=cfg.tag,
        use_relative_space=relative_space,
        variational=cfg.variational,
        num_anchor=cfg.num_anchor,
    )

if __name__ == "__main__":
    experiment()