import os

import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import html

TEXT_MODEL_NAME="all-MiniLM-L6-v2"

def load_model(model_name=TEXT_MODEL_NAME):
    model_path = f"models/{model_name}"
    if os.path.exists(model_path):
        print(f"Loading model from local path: {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"Model not found locally. Downloading and saving to: {model_path}")
        model = SentenceTransformer(f"sentence-transformers/{model_name}")
        model.save(model_path)
    return model


def extract_product_info(products_metadata, product_code, color_code):
    hits = products_metadata.get("hits", {}).get("hits", [])
    # Iterate through hits to find matching product
    for item in hits:
        source = item.get("_source", {})
        fields = item.get("fields", {})

        if source.get("baseProductCode") == product_code and source.get("colorCode") == color_code:

            cats = source.get("searchableCategoryNames", [])
            searchableCategoryNames = " ".join(html.unescape(item) for item in cats) if cats else ""

            searchablePatterns = ""
            patterns = fields.get("patterns", [])
            for pat in patterns:
                if pat:
                    searchablePatterns += html.unescape(pat)
                    searchablePatterns += " "

            result = f"{source.get("displayColorName")} {source.get("labelName")} {source.get("name")} " + \
                f"{source.get("filterData", {}).get("colorName")} {searchableCategoryNames} " + \
                f"{searchablePatterns}"
            return result

    return None  # If no match found


target="women_bekleidung"

metadata_path = f"input/{target}.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)


# Load index and IDs
index = faiss.read_index(f"faiss_{target}_index.bin")
with open(f"product_{target}_ids.json") as f:
    product_ids = json.load(f)

text_model = load_model()

prompt="blue sporty shirts from adidas"
query_emb = text_model.encode(prompt)

# Pad query embedding to match index dimension
# If you concatenated image+text, you need to pad zeros for image part:
query_vector = np.concatenate([np.zeros(1000), query_emb]).astype("float32").reshape(1, -1)

print(f"Results for prompt '{prompt}' taken from '{target}' category")
D, I = index.search(query_vector, 5)
for rank, idx in enumerate(I[0]):
    base = product_ids[idx].split(".")[0]
    product_code, color_code = base.split("_")
    print(f"Rank {rank+1}: Product ID = {product_ids[idx]}, Distance = {D[0][rank]}")
    print(f"Metadata: {extract_product_info(metadata, product_code, color_code)}")
