import pandas as pd
import re
import string

# load the data
df = pd.read_csv('bestsecret_reviews.csv')

# Проверка на пропущенные значения (если есть пустые отзывы - удаляем их)
df.dropna(subset=['content'], inplace=True)
df.reset_index(drop=True, inplace=True)

def clean_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление ссылок (если они есть)
    text = re.sub(r'http\S+', '', text)
    # Удаление знаков пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Удаление цифр (если они не несут смысла, но в отзывах могут быть важны)
    # text = re.sub(r'\d+', '', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Применяем очистку к столбцу с отзывами
df['cleaned_content'] = df['content'].apply(clean_text)

print("Очистка текста завершена. Первые 5 очищенных отзывов:")
print(df[['content', 'cleaned_content']].head())


# 1. Создаем бинарную метку
df['label'] = df['score'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))

# 2. Удаляем нейтральные (3-звездочные) отзывы для чистого бинарного анализа
df_labeled = df.dropna(subset=['label']).copy()

# 3. Переводим метку в строковый формат для удобства
df_labeled['sentiment'] = df_labeled['label'].apply(lambda x: 'POSITIVE' if x == 1 else 'NEGATIVE')

# 4. Проверяем распределение
sentiment_counts = df_labeled['sentiment'].value_counts()

print("\nРаспределение тональности по stars рейтингу:")
print(sentiment_counts)
print(f"\nИтоговый размер набора данных для анализа: {len(df_labeled)}")

# Сохраняем готовый набор данных
df_labeled[['cleaned_content', 'score', 'sentiment']].to_csv('bestsecret_preprocessed.csv', index=False)