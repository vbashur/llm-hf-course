from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch, json, os

# TARGET = "articleType"
TARGET = "baseColour"
MODEL_DIR = "./vit_" + TARGET

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, "id2label.json")) as f:
    id2label = json.load(f)

image_path = "kaggle-small/images/1557.jpg"
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
print("Predicted:", id2label[str(pred)])

image_path = "kaggle-small/images/1567.jpg"
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
print("Predicted:", id2label[str(pred)])
