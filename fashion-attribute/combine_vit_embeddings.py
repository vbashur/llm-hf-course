import numpy as np

# Load embeddings from both models
emb_article = np.load("embeddings_articleType.npy")
emb_color = np.load("embeddings_baseColour.npy")

# Check that both have the same number of samples
assert emb_article.shape[0] == emb_color.shape[0], "Number of samples must match!"

# Option 1️⃣: Concatenate embeddings (most common)
combined_embeddings = np.concatenate((emb_article, emb_color), axis=1)

# Option 2️⃣: Average embeddings (only if they have the same dimension)
# combined_embeddings = (emb_article + emb_color) / 2

# Save combined embeddings
np.save("embeddings_image_combined.npy", combined_embeddings)

print(f"Combined embeddings shape: {combined_embeddings.shape}")
