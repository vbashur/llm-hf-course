import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Load your embeddings
emb_text = np.load("embeddings_text.npy")
emb_image = np.load("embeddings_combined.npy")

# Reduce both to same dimension using PCA
shared_dim = 384
pca_text = PCA(n_components=shared_dim)
pca_img = PCA(n_components=shared_dim)

text_proj = pca_text.fit_transform(emb_text)
img_proj = pca_img.fit_transform(emb_image)

# Normalize for cosine similarity
text_proj = normalize(text_proj)
img_proj = normalize(img_proj)

np.save("embeddings_text_proj.npy", text_proj)
np.save("embeddings_image_proj.npy", img_proj)

print("Projected text:", text_proj.shape)
print("Projected image:", img_proj.shape)

