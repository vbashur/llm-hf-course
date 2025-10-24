import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---- Load projected embeddings ----
# Load embeddings from both models
emb_vit_article = np.load("embeddings_articleType.npy")
emb_vit_color = np.load("embeddings_baseColour.npy")

emb_text_all = np.load("embeddings_text_final_descriptions.npy")
meta = pd.read_csv("fashion_captions_combined.csv")


print("Text embeddings:", emb_text_all.shape)
print("Image article type embeddings:", emb_vit_article.shape)
print("Image color embeddings:", emb_vit_color.shape)


# # ---- Normalize for cosine similarity ----
# emb_img = normalize(emb_img)
# emb_text_all = normalize(emb_text_all)

# ---- Load same text model used before (DistilBERT) ----
category_model_name = "./fashion_category_classifier"
catagory_model_tokenizer = AutoTokenizer.from_pretrained(category_model_name)
category_classifier_model = AutoModelForSequenceClassification.from_pretrained(category_model_name)

color_model_name = "./fashion_color_classifier"
color_model_tokenizer = AutoTokenizer.from_pretrained(color_model_name)
color_classifier_model = AutoModelForSequenceClassification.from_pretrained(color_model_name)


def encode_text(text, text_model, tokenizer):
    base_model = text_model.base_model
    base_model.eval()
    """Convert text query into embedding vector using the fine-tuned text model"""
    with torch.no_grad():
        inputs = tokenizer(
            text, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        outputs = base_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding

# ---- Example query ----
query = "blue outdoor tunic"

query_emb_article = encode_text([query], category_classifier_model, catagory_model_tokenizer)
query_emb_color = encode_text([query], color_classifier_model, color_model_tokenizer)
# np.save("tmp_query_embeddings.npy", query_emb)
# emb_text = np.load("tmp_query_embeddings.npy")

print(f"Query embeddings shape: {query_emb_article.shape}")


# Compute similarities separately
# sim_color = cosine_similarity(query_emb_color, emb_vit_color)[0]
sim_type  = cosine_similarity(query_emb_article, emb_vit_article)[0]

similarities = sim_type # let's just check it for type only

# Combine with adjustable weights
# alpha = 0.5  # 0.5 means equal importance
# similarities = alpha * sim_color + (1 - alpha) * sim_type

# ---- Compute similarity ----
# similarities = cosine_similarity(query_emb, emb_text_all)[0]

# ---- Get top-N similar images ----
top_n = 5
top_idx = similarities.argsort()[-top_n:][::-1]

# ---- Show results ----
print("Query:", query)
print("\nTop matches:")
for i in top_idx:
    print(f"{i}: {meta.iloc[i]['productDisplayName']} | {meta.iloc[i]['articleType']} | {meta.iloc[i]['baseColour']}")
    print(f"Image path: {meta.iloc[i]['id']}")
    print("Similarity:", similarities[i])
    print("-" * 60)
