import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import json

# Load label mappings
with open("label_map.json") as f:
    id2label = json.load(f)
label2id = {v: k for k, v in id2label.items()}

# Load dataset
df = pd.read_csv("fashion_captions_combined.csv")
df["text"] = df["final_description"].fillna("") + " " + df["generated_caption"].fillna("")
df = df.dropna(subset=["articleType"])
df["label"] = df["articleType"].astype("category").cat.codes

# Train/test split
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Model name — small but effective multilingual model
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

class FashionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = FashionDataset(train_df["text"].tolist(), train_df["label"].tolist())
val_dataset = FashionDataset(val_df["text"].tolist(), val_df["label"].tolist())

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
)

training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./fashion_category_classifier")
tokenizer.save_pretrained("./fashion_category_classifier")
