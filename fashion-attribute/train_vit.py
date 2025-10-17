import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer

# ========== CONFIG ==========
# TARGET_COL = "articleType"   # or 'baseColour'
TARGET_COL = "baseColour"
CSV_PATH = "fashion_captions_combined.csv"
IMAGES_DIR = "kaggle-small/images"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
OUTPUT_DIR = "./vit_" + TARGET_COL
EPOCHS = 5
BATCH_SIZE = 8
# ============================

# Load dataset
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET_COL])
df["image_path"] = df["id"].astype(str) + ".jpg"

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df[TARGET_COL])
num_labels = len(le.classes_)
print(f"Training ViT on {num_labels} classes ({TARGET_COL})")

# Save label map
label2id = {label: i for i, label in enumerate(le.classes_)}
id2label = {i: label for label, i in label2id.items()}

# Split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Processor for image transformations
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

class FashionImageDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Skip broken images
            return self.__getitem__((idx + 1) % len(self.df))

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(row["label"])
        return inputs

train_dataset = FashionImageDataset(train_df, IMAGES_DIR, processor)
val_dataset = FashionImageDataset(val_df, IMAGES_DIR, processor)

# Load pre-trained ViT
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# Training setup
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
)

trainer.train()

# Save model and label mappings
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

import json
with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w") as f:
    json.dump(id2label, f)
