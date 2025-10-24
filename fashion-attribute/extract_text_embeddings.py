# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd, numpy as np
import torch

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model = SentenceTransformer("./fashion_category_classifier")
model_name = "./fashion_category_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
base_model = model.base_model  # this gives you the encoder part


df = pd.read_csv("fashion_captions_combined.csv")
texts = df["final_description"].tolist()

embeddings = []
base_model.eval()
with torch.no_grad():
    for i in range(0, len(texts), 32):  # batch size 32
        batch = texts[i:i+32]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        outputs = base_model(**inputs)
        # use the [CLS] token representation
        # with torch.no_grad():
        #     outputs = model(**inputs)
        # emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        # cls_embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

# 4️⃣ Combine all batches
embeddings = np.concatenate(embeddings, axis=0)
embeddings_stack = np.stack(embeddings)
print("Embeddings shape:", embeddings_stack.shape)

# 5️⃣ Save to .npy file
np.save("embeddings_text_final_descriptions.npy", embeddings_stack)
print("✅ Saved text embeddings to embeddings_text_final_descriptions.npy")

