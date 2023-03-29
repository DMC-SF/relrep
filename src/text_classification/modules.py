from typing import Optional, Dict

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

    def forward(self, inputs: Dict) -> torch.Tensor:
        with torch.no_grad():
            x = self.encoder(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        x = self.decoder(x.pooler_output)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.cross_entropy(y_hat, batch["labels"])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.cross_entropy(y_hat, batch["labels"])
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class AmazonReviewsDataModule(pl.LightningDataModule):
    loader_columns = {
        'roberta-base': ["input_ids", "attention_mask", "labels",],
        'bert-base': ["input_ids", "token_type_ids", "attention_mask", "labels",]
    }

    def __init__(self, model_name: str = "roberta-base", language: str = "en", batch_size: int = 128):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        self.dataset = load_dataset("amazon_reviews_multi", self.language)
        self.dataset = self.dataset.rename_column("stars", "labels")
        self.dataset = self.dataset.map(self.tokenize, batched=True)
        self.dataset = self.dataset.map(lambda batch: {"labels": batch["labels"] - 1})
        self.dataset.set_format(type="torch", columns=self.loader_columns[self.model_name])
        self.train_dataset, self.val_dataset = self.dataset["train"], self.dataset["test"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def tokenize(self, batch):
        """Tokenize and add labels to the batch."""
        return self.tokenizer(batch["review_body"], padding=True, truncation=True, return_tensors="pt")
