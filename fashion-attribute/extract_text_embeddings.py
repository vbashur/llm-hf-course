from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
df = pd.read_csv("fashion_captions_combined.csv")

texts = (df["final_description"].fillna("") + " " + df["generated_caption"].fillna("")).tolist()
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

np.save("embeddings_text.npy", embeddings)
df[["id"]].to_csv("text_embedding_ids.csv", index=False)
print(f"text embeddings shape {embeddings.shape}")
