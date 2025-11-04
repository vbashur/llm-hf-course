import os
import torch
from PIL import Image
from torchvision import models, transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import numpy as np
import json
import re

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ViT model for embeddings
vit_model = models.vit_b_16(pretrained=True).to(device)
vit_model.eval()

# Load BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Image preprocessing for ViT
vit_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Folder containing images
# target="women_bekleidung"
target="men_bekleidung"
image_folder = f"./input/{target}"

# Output dictionaries
embeddings = {}
descriptions = []

def remove_repeats(text):
    # Remove consecutive duplicate words
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

# Process images
for image_name in tqdm(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_name)
    try:
        image = Image.open(image_path).convert("RGB")

        # Generate embedding
        input_tensor = vit_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = vit_model(input_tensor)
        embeddings[image_name] = embedding.squeeze().cpu().numpy().tolist()

        # Generate description

        # inputs = processor(image, return_tensors="pt").to(device)
        # out = model.generate(**inputs, max_length=30)
        # caption = processor.decode(out[0], skip_special_tokens=True)
        #
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_length=30)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        # descriptions[image_name] = remove_repeats(description)
        cleaned_desc = remove_repeats(description)
        (productCode, colorCode) = image_name.replace('.jpg', '').split('_', 1)
        descriptions.append({
            "baseProductCode": productCode.lstrip('0'),
            "colorCode": colorCode,
            "description": cleaned_desc
        })

    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# Save results
with open(f"./generated/image_{target}_embeddings.json", "w") as f:
    json.dump(embeddings, f)

result_descriptions = {"descriptions": descriptions}
with open(f"./generated/image_{target}_descriptions.json", "w") as f:
    json.dump(result_descriptions, f)
