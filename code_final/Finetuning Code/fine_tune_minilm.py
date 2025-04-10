import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Set device: use GPU if available, otherwise CPU
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load Financial PhraseBank dataset from data.csv
# ----------------------------
df = pd.read_csv("data.csv")  # Assumes file with columns: Sentence and Sentiment
df['Sentence'] = df['Sentence'].astype(str)

# Map sentiment labels to integers (e.g., negative=0, neutral=1, positive=2)
mapping = {"negative": 0, "neutral": 1, "positive": 2}
if df['Sentiment'].dtype == object:
    df['Sentiment'] = df['Sentiment'].map(mapping)

# ----------------------------
# Compute class weights to address imbalance
# ----------------------------
classes = np.unique(df['Sentiment'])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=df['Sentiment'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# ----------------------------
# Define a Dataset class for Financial PhraseBank
# ----------------------------
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
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove extra batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

# ----------------------------
# Load the tokenizer for MiniLM
# ----------------------------
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
texts = df['Sentence'].tolist()
labels = df['Sentiment'].tolist()
dataset = FinancialPhraseBankDataset(texts, labels, tokenizer, max_length=128)

# ----------------------------
# Define a custom model to include weighted loss
# ----------------------------
class WeightedModelForSequenceClassification(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.class_weights = None  # Initialize without weights

    def forward(self, **kwargs):
        labels = kwargs.get("labels")
        outputs = super().forward(**kwargs)
        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(outputs.logits, labels)
            return {"loss": loss, "logits": outputs.logits}
        return outputs

# ----------------------------
# Load model configuration and instantiate the model
# ----------------------------
num_labels = len(classes)  # e.g., 3 for negative, neutral, positive
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model = WeightedModelForSequenceClassification.from_pretrained(model_name, config=config)
model.class_weights = class_weights  # Set class weights after loading
model.to(device)

# ----------------------------
# Define compute_metrics function for evaluation
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ----------------------------
# Set training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir='./results_finminilm_finphrasebank',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    logging_dir='./logs_finminilm_finphrasebank',
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# ----------------------------
# Instantiate the Trainer and fine-tune the model
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./minilm_finphrasebank_final_model")
