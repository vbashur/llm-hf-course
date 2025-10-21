import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ---- Load projected embeddings ----
emb_img = np.load("embeddings_image_proj.npy")
emb_text_all = np.load("embeddings_text_proj.npy")
meta = pd.read_csv("fashion_captions_combined.csv")



# # ---- Normalize for cosine similarity ----
# emb_img = normalize(emb_img)
# emb_text_all = normalize(emb_text_all)

# ---- Load same text model used before (DistilBERT) ----
# text_model = SentenceTransformer("distilbert-base-uncased")
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---- Example query ----
query = "red sporty sneakers"

query_emb = text_model.encode([query], show_progress_bar=True, convert_to_numpy=True)
np.save("tmp_query_embeddings.npy", query_emb)
emb_text = np.load("tmp_query_embeddings.npy")

# print(f"Query embeddings shape: {query_emb.shape}")
# shared_dim = 384
# pca_text = PCA(n_components=shared_dim)
#
# text_proj = pca_text.fit_transform(emb_text)

query_emb_proj = normalize(emb_text)
emb_text_all = normalize(emb_text_all)


# ---- Compute similarity ----
similarities = cosine_similarity(query_emb_proj, emb_text_all)[0]

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
