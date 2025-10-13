import pandas as pd
import re


def clean_caption(caption: str) -> str:
    if not isinstance(caption, str):
        return ""
    caption = caption.lower()
    # Remove repetitive brand words (like "adidas adidas adidas")
    caption = re.sub(r'\b(\w+)( \1\b)+', r'\1', caption)
    # Remove person descriptors
    caption = re.sub(r'\ba (man|woman|girl|boy) (wearing|in)\b', '', caption)
    # Clean up whitespace and trailing periods
    caption = re.sub(r'\s+', ' ', caption).strip().capitalize()
    return caption

def combine_descriptions(row):
    orig = str(row.get('productDisplayName', '')).strip()
    gen = str(row.get('generated_caption_clean', '')).strip()
    if not orig:
        return gen
    if not gen:
        return orig
    # Example rule: prioritize Kaggle’s factual info + natural caption flavor
    return f"{orig}. {gen[0].upper() + gen[1:] if len(gen) > 1 else gen}"

# Kaggle dataset (original)
df_kaggle = pd.read_csv("kaggle-small/styles.csv", on_bad_lines='skip', engine='python')

# Generated captions
df_gen = pd.read_csv("generated_captions.csv")

print(len(df_kaggle), len(df_gen))

df_gen['id'] = df_gen['image'].astype(str)
df_kaggle['id'] = df_kaggle['id'].astype(str)

df = pd.merge(df_kaggle, df_gen, on="id", how="inner")
print("Merged rows:", len(df))

df['generated_caption_clean'] = df['generated_caption'].apply(clean_caption)
df['final_description'] = df.apply(combine_descriptions, axis=1)

df[['id', 'final_description', 'articleType','baseColour','gender','season', 'productDisplayName', 'generated_caption']].to_csv(
    "fashion_captions_combined.csv",
    index=False
)