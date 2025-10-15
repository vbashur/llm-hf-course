from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json

model_name = "./fashion_color_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

with open("color_map.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

text = "a blue slim fit pants for men"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
print(f"Original text: {text}")
print("Predicted color:", id2label[pred])

text = "a dark red leather jacket for women"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
print(f"Original text: {text}")
print("Predicted color:", id2label[pred])
