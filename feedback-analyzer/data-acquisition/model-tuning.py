import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

# --- 1. Constants ---
MODEL_CHECKPOINT = "siebert/sentiment-roberta-large-english"
NUM_LABELS = 2  # Positive and Negative
OUTPUT_DIR = "./results/sentiment_finetuning"
EPOCHS = 3

# --- 2. Load and Prepare Data ---
df = pd.read_csv('bestsecret_preprocessed.csv')

# The Trainer expects columns named 'text' and 'label' (as int)
df = df.rename(columns={'cleaned_content': 'text', 'score': 'label'})

# Convert pandas DataFrame to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df[['text', 'label']])

# Split the dataset (80% train, 20% test/evaluation)
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

tokenized_datasets = train_test_split.map(
    lambda examples: tokenizer(examples["text"], truncation=True),
    batched=True
)

print(f"Размер обучающего набора: {len(tokenized_datasets['train'])} отзывов")
print(f"Размер оценочного набора: {len(tokenized_datasets['test'])} отзывов")

# --- 3. Load Model and Tokenizer ---

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS
)

# Define label mapping for clarity (0=Negative, 1=Positive)
model.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
model.config.label2id = {'NEGATIVE': 0, 'POSITIVE': 1}

# --- 4. Define Metrics ---
metric = evaluate.load("f1")  # We'll use F1-score, common for classification


def compute_metrics(eval_pred):
    # This function is called after each evaluation step
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate F1-score for binary classification
    f1_score = metric.compute(predictions=predictions, references=labels, average='weighted')

    # Calculate simple accuracy
    accuracy = (predictions == labels).mean()

    return {"f1": f1_score['f1'], "accuracy": accuracy}


# --- 5. Configure Training Arguments and Trainer ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16,  # Adjust based on your GPU/RAM
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",  # Save checkpoint after each epoch
    load_best_model_at_end=True,  # Load the best model found during evaluation
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 6. Train the Model ---
print("\n🚀 Начинаем Fine-tuning модели...")
trainer.train()

# --- 7. Evaluation ---
print("\n✅ Финальная оценка модели на тестовом наборе:")
results = trainer.evaluate()
print(results)