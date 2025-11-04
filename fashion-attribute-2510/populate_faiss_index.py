import json
from idlelib.colorizer import color_config

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import html
from tqdm import tqdm



def extract_generated_description(descriptions_data, product_code: str, color_code: str) -> str:
    for item in descriptions_data.get('descriptions', []):
        if item.get('baseProductCode') == product_code and item.get('colorCode') == color_code:
            return item.get('description')
    return None


def extract_product_info(products_metadata, product_code, color_code):

    # Navigate to hits
    hits = products_metadata.get("hits", {}).get("hits", [])

    # Iterate through hits to find matching product
    for item in hits:
        source = item.get("_source", {})
        fields = item.get("fields", {})

        if source.get("baseProductCode") == product_code and source.get("colorCode") == color_code:
            # Extract required fields
            # result = {
            #     "displayColorName": source.get("displayColorName"),
            #     "labelName": source.get("labelName"),
            #     "name": source.get("name"),
            #     "filterData.colorName": source.get("filterData", {}).get("colorName"),
            #     "searchableCategoryNames": source.get("searchableCategoryNames", []),
            #     "fields.patterns": fields.get("patterns", [])
            # }
            cats = source.get("searchableCategoryNames", [])
            searchableCategoryNames = " ".join(html.unescape(item) for item in cats) if cats else ""

            searchablePatterns = ""
            patterns = fields.get("patterns", [])
            for pat in patterns:
                if pat:
                    searchablePatterns += html.unescape(pat)
                    searchablePatterns += " "

            # searchablePatterns = " ".join(html.unescape(item) for item in patterns) if patterns else ""

            result = f"{source.get("displayColorName")} {source.get("labelName")} {source.get("name")} " + \
                f"{source.get("filterData", {}).get("colorName")} {searchableCategoryNames} " + \
                f"{searchablePatterns}"
            return result

    return None  # If no match found




target="women_bekleidung"
# Paths
image_embeddings_path = f"generated/image_{target}_embeddings.json"
descriptions_path = f"generated/image_{target}_descriptions.json"
metadata_path = f"input/{target}.json"

# Load data
with open(image_embeddings_path, "r") as f:
    image_embeddings = json.load(f)

with open(descriptions_path, "r") as f:
    descriptions = json.load(f)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Initialize text embedding model
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

combined_embeddings = []
product_ids = []

for product_id in tqdm(image_embeddings.keys()):

    img_emb = np.array(image_embeddings[product_id])

    base = product_id.split(".")[0]
    product_code, color_code = base.split("_")


    desc = extract_generated_description(descriptions, product_code, color_code)

    meta = extract_product_info(metadata, product_code, color_code)
    text_emb = text_model.encode(desc + " " + meta)

    # do normalization for both embeddings

    img_emb = img_emb / np.linalg.norm(img_emb)
    text_emb = text_emb / np.linalg.norm(text_emb)

    # Combine embeddings (simple average)
    combined = np.concatenate([img_emb, text_emb])
    combined_embeddings.append(combined)
    product_ids.append(product_id)

# Convert to NumPy array
combined_embeddings = np.vstack(combined_embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(combined_embeddings.shape[1])
index.add(combined_embeddings)

# Save index and mapping
faiss.write_index(index, f"faiss_{target}_index.bin")
with open(f"product_{target}_ids.json", "w") as f:
    json.dump(product_ids, f)

print(f"FAISS index built with {len(product_ids)} products for {target} category.")
