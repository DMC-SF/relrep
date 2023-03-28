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

from utils.utils import load_anchors, select_random_anchors, strip_and_load

from models.autoencoder import AutoEncoder


class MNISTModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_anchor: int,
        batch_size: int,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        # save network architecture
        self.net = net

        # save optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # loss
        self.loss = nn.MSELoss()

        # select of N anchors
        select_random_anchors(
            MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
            n_anchors=num_anchor,
            data_dir="data", 
            anchors_dir="anchors"
        )         
    
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
        dataset = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        dataset = MNIST("data", train=False, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)