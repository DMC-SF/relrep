import os

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from modules import AmazonReviewsDataModule, TextClassificationModule
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer


def train_encoder():
    pl.seed_everything(42)

    hparams = {
        "model_name": "roberta-base",
        "language": "en",
        "batch_size": 128,
        "learning_rate": 1e-5,
        "max_epochs": 10,
    }

    datamodule = AmazonReviewsDataModule(
        model_name=hparams["model_name"],
        language=hparams["language"],
        batch_size=hparams["batch_size"],
    )

    model = TextClassificationModule(
        encoder=AutoModel.from_pretrained("roberta-base"),
        num_classes=5,
        learning_rate=hparams["learning_rate"],
    )

    logger = pl.loggers.WandbLogger(
        project="cross-lingual-stitching",
        name=f"{hparams['model_name']}-{hparams['language']}",
        save_dir=os.getcwd(),
        offline=False,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=os.path.join(logger.experiment.dir, "checkpoints"),
            filename="{epoch:02d}-{val/loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=5, mode="min"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=[0],
        max_epochs=hparams["max_epochs"],
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_encoder()
