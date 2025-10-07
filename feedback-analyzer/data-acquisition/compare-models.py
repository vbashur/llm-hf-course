from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

test_reviews = [
        "I love this app, it's fast and easy to use!",  # Ожидаем POSITIVE
        "The shipping was terribly slow and the clothes were damaged. Bad experience.", # Ожидаем NEGATIVE
        "Bad", # Ваш пример, ожидаем NEGATIVE
        "This is a total disaster, everything about this update is awful.", # Ожидаем NEGATIVE
        "It's okay, nothing special, but it works.", # Ожидаем NEUTRAL (модель выдаст либо POS, либо NEG)
    ]

def original_model_check():
    # 1. Загрузка ЧИСТОЙ исходной модели с Hugging Face Hub
    BASE_MODEL = "siebert/sentiment-roberta-large-english"

    # Создаем пайплайн для анализа тональности
    try:
        # Используем CPU (-1) для быстрой проверки, если нет GPU
        base_sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=BASE_MODEL,
            device=-1
        )
    except Exception as e:
        print(f"Ошибка при загрузке базовой модели: {e}")
        print("Убедитесь, что установлены `transformers` и `torch`.")
        exit()

    # 2. Тестовые примеры


    print(f"\n--- Проверка Чистой Модели: {BASE_MODEL} ---")
    results = base_sentiment_pipeline(test_reviews)

    for review, result in zip(test_reviews, results):
        # Модель SieBERT возвращает 0 (NEGATIVE) или 1 (POSITIVE)
        label = "POSITIVE" if result['label'] == 1 else "NEGATIVE"
        print(f"\nОтзыв: '{review}'")
        print(f"Результат: {label} (Score: {result['score']:.4f})")

def original_model_grained_check():
    # 1. Загрузка компонентов
    MODEL_NAME = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # 2. Определение меток
    # SieBERT configuration: 0 -> NEGATIVE, 1 -> POSITIVE
    id_to_label = {0: 'NEGATIVE', 1: 'POSITIVE'}

    for review in test_reviews:
        # 3. Токенизация
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)

        # 4. Прогноз
        with torch.no_grad():
            outputs = model(**inputs)

        # Получение ID с наибольшей вероятностью
        predicted_id = torch.argmax(outputs.logits, dim=-1).item()

        # Получение текстовой метки и score
        predicted_label = id_to_label[predicted_id]

        # Применение softmax для получения вероятности (Score)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        score = probabilities[predicted_id].item()

        print(f"\nОтзыв: '{review}'")
        print(f"ID Прогноза: {predicted_id}")
        print(f"Результат: {predicted_label} (Score: {score:.4f})")



def tuned_model_check():
    # Замените этот путь на путь к лучшему чекпоинту, который вы определили
    BEST_CHECKPOINT_PATH = "./results/sentiment_finetuning/checkpoint-147"

    print(f"\n--- Проверка Дообученной Модели: {BEST_CHECKPOINT_PATH} ---")

    try:
        fine_tuned_sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=BEST_CHECKPOINT_PATH,
            tokenizer=BEST_CHECKPOINT_PATH,
            device=-1
        )

        results_ft = fine_tuned_sentiment_pipeline(test_reviews)

        for review, result in zip(test_reviews, results_ft):
            # Модель SieBERT возвращает 0 (NEGATIVE) или 1 (POSITIVE)
            label = "POSITIVE" if result['label'] == 1 else "NEGATIVE"
            print(f"\nОтзыв: '{review}'")
            print(f"Результат (FT): {label} (Score: {result['score']:.4f})")

    except Exception as e:
        print(f"Ошибка при загрузке дообученной модели: {e}")


print(f"Check the original model in grained mode")
original_model_grained_check()
print(f"Check the original model in regular mode")
original_model_check()
# tuned_model_check()
