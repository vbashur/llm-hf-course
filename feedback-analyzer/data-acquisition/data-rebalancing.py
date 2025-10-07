import pandas as pd
from datasets import Dataset

# Загрузка и разделение данных
df_labeled = pd.read_csv('bestsecret_preprocessed.csv')
df = df_labeled.rename(columns={'cleaned_content': 'text', 'score': 'label'})

# 1. Разделение на классы
df_positive = df[df['label'] == 5]
df_negative = df[df['label'] == 1]

# 2. Определение размера меньшего класса (например, NEGATIVE)
min_size = len(df_negative)

# 3. Уменьшение доминирующего класса (POSITIVE) до min_size
df_positive_downsampled = df_positive.sample(n=min_size, random_state=42)

# 4. Объединение сбалансированных данных
df_balanced = pd.concat([df_positive_downsampled, df_negative])

# 5. Перемешивание
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Новое распределение классов: {df_balanced['label'].value_counts()}")

# Сохранение сбалансированного набора
df_balanced.to_csv('bestsecret_balanced.csv', index=False)