from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import torch, os, pandas as pd, numpy as np
from tqdm import tqdm

# TARGET="articleType" # or baseColour
TARGET="baseColour"
MODEL_DIR = "./vit_" + TARGET
CSV_PATH = "fashion_captions_combined.csv"
IMAGES_DIR = "kaggle-small/images"

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTModel.from_pretrained(MODEL_DIR)
model.eval()

df = pd.read_csv(CSV_PATH)
embeddings, ids = [], []

for i, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMAGES_DIR, str(row["id"]) + ".jpg")
    if not os.path.exists(img_path):
        continue
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(emb)
        ids.append(row["id"])
    except:
        continue
TARGET_EMBEDDINGS_FILENAME = f"embeddings_{TARGET}.npy"
np.save(TARGET_EMBEDDINGS_FILENAME, np.stack(embeddings))

TARGET_DATAFRAME_FILENAME = f"vit_ids_{TARGET}.csv"
pd.DataFrame({"id": ids}).to_csv(TARGET_DATAFRAME_FILENAME, index=False)
