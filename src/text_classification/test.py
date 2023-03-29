import os

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Load the Amazon Reviews dataset
dataset = load_dataset("amazon_reviews_multi", "en")
dataset = dataset.rename_column("stars", "labels")


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Apply the tokenizer to the dataset


def tokenize_function(batch):
    """Tokenize."""
    return tokenizer(batch["review_body"], padding=True, truncation=True, return_tensors="pt")


dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(lambda batch: {"labels": batch["labels"] - 1})

loader_columns = [
    "input_ids",
    "attention_mask",
    "labels",
]
dataset.set_format(type="torch", columns=loader_columns)

# Split the dataset into train and test sets
train_data = dataset["train"]
test_data = dataset["test"]

# Create a dataloader and print a batch
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)

for i, batch in enumerate(train_dataloader):
    if i == 10:
        break

# Create a model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

# Pass the input to the model
output = model(batch["input_ids"], attention_mask=batch["attention_mask"])
print(output.keys())