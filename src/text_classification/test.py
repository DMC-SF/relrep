import os

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoTokenizer

# Load the Amazon Reviews dataset
dataset = load_dataset("amazon_reviews_multi", "en")
dataset = dataset.rename_column("stars", "labels")


# Split the dataset into train and test sets
train_data = dataset["train"]
test_data = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Apply the tokenizer to the dataset


def tokenize_function(batch):
    """Tokenize."""
    return tokenizer(batch["review_body"], padding=True, truncation=True, return_tensors="pt")


tokenized_train_data = train_data.map(tokenize_function, batched=True)
tokenized_train_data = tokenized_train_data.map(lambda batch: {"labels": batch["labels"] - 1})

loader_columns = [
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "labels",
]
tokenized_train_data.set_format(type="torch", columns=loader_columns)

# Create a dataloader and print a batch
train_dataloader = torch.utils.data.DataLoader(tokenized_train_data, batch_size=2)

for batch in train_dataloader:
    break

# Create a model
model = AutoModel.from_pretrained("roberta-base")

# Pass the input to the model
output = model(**batch)
