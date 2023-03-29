import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from utils.utils import save_anchors, load_anchors

from models.autoencoder import AutoEncoder


class MNISTModule(LightningModule):
    def __init__(
        self,
        # net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_anchor: int,
        batch_size: int,
        layer_size: int,
        hidden_size: int,
        use_relative_space: bool,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        # loss
        self.loss = nn.MSELoss()

        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        self.dataset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))

        # select of N anchors
        save_anchors(
            self.dataset,
            n_anchors=num_anchor
        ) 
        self.anchors = load_anchors()[:num_anchor, :, :, :]

        # save network architecture
        self.net = AutoEncoder(
            anchors=self.anchors,
            layer_size = layer_size, 
            hidden_size=hidden_size, 
            use_relative_space=use_relative_space
        )

        # save optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler        
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        y = self(x)
        loss = self.loss(y, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        y = self(x)
        loss = self.loss(y, x)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=False)