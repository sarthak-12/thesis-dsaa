#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import warnings
import random

warnings.filterwarnings("ignore")

# ----------------------------------
# 1. Reproducibility + Device
# ----------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------------
# 2. Load & Preprocess Financial PhraseBank
# ----------------------------------
df = pd.read_csv("data.csv")  # Has columns: ["Sentence", "Sentiment"]
df["Sentence"] = df["Sentence"].astype(str)

mapping = {"negative": 0, "neutral": 1, "positive": 2}
if df["Sentiment"].dtype == object:
    df["Sentiment"] = df["Sentiment"].map(mapping)

classes = np.unique(df["Sentiment"])
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=df["Sentiment"])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# Simple train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Sentence"].tolist(),
    df["Sentiment"].tolist(),
    test_size=0.2,
    stratify=df["Sentiment"],
    random_state=seed
)

# ----------------------------------
# 3. Create a Dataset Class
# ----------------------------------
class FinancialPhraseBankDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove extra batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = FinancialPhraseBankDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset   = FinancialPhraseBankDataset(val_texts,   val_labels,   tokenizer, max_length=128)

# ----------------------------------
# 4. Weighted Model Definition
# ----------------------------------
class WeightedModelForSequenceClassification(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.class_weights = None

    def forward(self, **kwargs):
        labels = kwargs.get("labels")
        outputs = super().forward(**kwargs)
        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(outputs.logits, labels)
            return {"loss": loss, "logits": outputs.logits}
        return outputs

num_labels = len(classes)
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
# Optionally increase dropout
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

model = WeightedModelForSequenceClassification.from_pretrained(model_name, config=config)
model.class_weights = class_weights
model.to(device)

# ----------------------------------
# 5. Metrics + Training Args
# ----------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results_deberta_v3_finphrasebank",
    num_train_epochs=3,          
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs_deberta_v3_finphrasebank",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    warmup_steps=500,
    max_grad_norm=0.5,
    fp16=True,
    lr_scheduler_type="linear",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# ----------------------------------
# 6. Fine-tune and Save
# ----------------------------------
trainer.train()
trainer.save_model("./deberta_v3_finphrasebank_final_model")
print("Finetuning complete. Saved to ./deberta_v3_finphrasebank_final_model")
