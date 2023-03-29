import os

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from modules import AmazonReviewsDataModule, TextClassificationModule

loader_columns = [
    "input_ids",
    "attention_mask",
    "labels",
]

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def tokenize_function(batch):
    """Tokenize."""
    return tokenizer(batch["review_body"], max_length=512, padding="max_length", truncation=True, return_tensors="pt")


dataset = load_dataset("amazon_reviews_multi", "en")
dataset = dataset.rename_column("stars", "labels")
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(lambda batch: {"labels": batch["labels"] - 1})
dataset.set_format(type="torch", columns=loader_columns)
train_data, val_data = dataset["train"], dataset["test"]

# Create a dataloader and print a batch
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)

for i, batch in enumerate(train_dataloader):
    print(batch["input_ids"].shape)

print("OK1")

val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

if 1:
    datamodule = AmazonReviewsDataModule(
        model_name="roberta-base",
        language="en",
        batch_size=256,
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        print(batch["input_ids"].shape)

    val_loader = datamodule.val_dataloader()

    for batch in val_loader:
        print(batch["input_ids"].shape)