from transformers import pipeline

# Замените 'checkpoint-147' на фактическую лучшую папку, которую вы определили
BEST_CHECKPOINT_PATH = "./results/sentiment_finetuning"

# Загружаем модель для инференса
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=BEST_CHECKPOINT_PATH,
    tokenizer=BEST_CHECKPOINT_PATH,
    device=0 # device=0 для GPU, device=-1 для CPU
)

# Проверяем на тестовом примере
print(sentiment_pipeline("absolutely bad"))