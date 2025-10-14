import pandas as pd
from sklearn.model_selection import train_test_split

# Load your combined dataset
df = pd.read_csv("fashion_captions_combined.csv")

# Combine all textual info into a single input field
df["text"] = df["final_description"].fillna("") + " " + df["generated_caption"].fillna("")

# Choose one or more attributes to predict
target_col = "articleType"  # or try "baseColour" or "gender"

# Clean and encode labels
df = df.dropna(subset=[target_col])
labels = df[target_col].astype("category").cat.codes
label2id = dict(enumerate(df[target_col].astype("category").cat.categories))
id2label = {v: k for k, v in label2id.items()}

# Split train/test
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    labels.tolist(),
    test_size=0.2,
    random_state=42,
)

print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

# Optionally save mappings
import json
with open("label_map.json", "w") as f:
    json.dump(id2label, f)
