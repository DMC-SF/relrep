from typing import Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class TextClassificationModule(pl.LightningModule):
    def __init__(
        self,
        encoder: AutoModel,
        num_classes: int,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "encoder"])
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.config.hidden_size, num_classes)

    def forward(self, **inputs) -> torch.Tensor:
        with torch.no_grad():
            x = self.encoder(**inputs)
        x = self.decoder(x.last_hidden_state[:, 0])
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss = F.cross_entropy(y_hat, batch["stars"] - 1)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss = F.cross_entropy(y_hat, batch["stars"] - 1)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class AmazonReviewsDataModule(pl.LightningDataModule):
    loader_columns = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
    ]

    def __init__(self, model_name: str, language: str, batch_size: int = 128):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        self.dataset = load_dataset("amazon_reviews_multi", self.language)
        self.dataset = self.dataset.map(self.tokenize, batched=True)
        self.dataset.set_format(type="torch", columns=self.loader_columns)
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["test"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def process_batch(self, batch):
        """Tokenize and add labels to the batch."""
        batch["input_ids"] = self.tokenizer(
            batch["review_body"],
            padding=True,
            truncation=True,
        )["input_ids"]
        batch["labals"] = batch["stars"].astype(torch.long) - 1
        return batch
