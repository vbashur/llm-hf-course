from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, os, pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"get the model")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

print(f"read dataset")
df = pd.read_csv("kaggle-small/styles.csv", on_bad_lines='skip', engine='python')   # Kaggle Fashion Product Images (Small)
df = df.dropna(subset=['id', 'productDisplayName'])  # drop empty critical fields
df = df.reset_index(drop=True)

image_dir = "kaggle-small/images/"

results = []
head_count=10000
print(f"handle {head_count} rows")
for i, row in df.head(head_count).iterrows():
    img_path = os.path.join(image_dir, str(row['id']) + ".jpg")
    try:
        image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image not found: {img_path}, skipping.")
        continue

    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=30)
    caption = processor.decode(out[0], skip_special_tokens=True)

    results.append({"image": row['id'], "generated_caption": caption})

genrated_captions_filename = "generated_captions.csv"
print(f"save data to {genrated_captions_filename}")
pd.DataFrame(results).to_csv(genrated_captions_filename, index=False)

