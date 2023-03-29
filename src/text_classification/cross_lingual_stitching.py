import os

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from modules import AmazonReviewsDataModule, TextClassificationModule
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer

# Set the random seed for reproducibility
torch.manual_seed(42)


def train_encoder():
    datamodule = AmazonReviewsDataModule(
        model_name="roberta-base",
        language="en",
        batch_size=64,
    )

    model = TextClassificationModule(
        encoder=AutoModel.from_pretrained("roberta-base"),
        num_classes=5,
        learning_rate=1e-3,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1],
        max_epochs=3,
        logger=False,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_encoder()
