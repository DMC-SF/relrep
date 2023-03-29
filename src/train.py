import logging
import warnings
from pytorch_lightning import seed_everything

import hydra
import os
import torch

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

log = logging.getLogger(__name__)
os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    seed_everything(cfg.seed, workers=True)

    logger = hydra.utils.instantiate(cfg.logger)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model=model)

    if not os.path.exists("weights"):
        os.makedirs("weights")

    use_relative_space = cfg.model.net.use_relative_space

    print("Saving weights...")
    print("Encoder weights...")
    save_path = f"weights/enc_seed={cfg.seed}_relative_space={use_relative_space}.pt"
    torch.save(model.net.encoder.state_dict(), save_path)
    print("Decoder weights...")
    save_path = f"weights/dec_seed={cfg.seed}_relative_space={use_relative_space}.pt"
    torch.save(model.net.decoder.state_dict(), save_path)

if __name__ == "__main__":
    main()