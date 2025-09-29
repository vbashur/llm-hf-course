import pandas as pd
from google_play_scraper import Sort, reviews

# 1. Application ID
APP_ID = 'com.bestsecret.main'
# 2. Feedback language (en)
LANGUAGE = 'en'
# 3. Number of feedbacks
COUNT = 1000

print(f"Start collecting feedbacks {APP_ID}...")

# Сбор отзывов с использованием `reviews`
result, continuation_token = reviews(
    APP_ID,
    lang=LANGUAGE,
    country='de',  # Target country
    sort=Sort.NEWEST,  # Sort by most recent
    count=COUNT,       # Desired number of reviews
    filter_score_with=None # All the scores (1-5 stars)
)

print(f"Collected {len(result)} feedbacks.")

# Преобразование в DataFrame для удобства
df = pd.DataFrame(result)

# Отбираем только нужные столбцы: 'content' (текст отзыва) и 'at' (дата)
# Также сохраним 'score' (звездный рейтинг) для дальнейшего сравнения с моделью
df_cleaned = df[['userName', 'content', 'score', 'at', 'thumbsUpCount']]

# Переименуем столбец 'at' в 'date'
df_cleaned.rename(columns={'at': 'date'}, inplace=True)

# Сохранение в CSV-файл
CSV_FILENAME = 'bestsecret_reviews.csv'
df_cleaned.to_csv(CSV_FILENAME, index=False)

print(f"Successully collected into the file: {CSV_FILENAME}")
print("\nTop 5 lines of collected reviews:")
print(df_cleaned.head())