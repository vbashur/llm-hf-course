from sklearn.metrics.pairwise import cosine_similarity
import numpy as np, pandas as pd

img_emb = np.load("embeddings_image_proj.npy")
txt_emb = np.load("embeddings_text_proj.npy")

# Example: find top-5 most similar images to the first text
sims = cosine_similarity(txt_emb[0:1], img_emb)[0]
top_idx = sims.argsort()[-5:][::-1]
print("Top-5 similar image indices:", top_idx)
