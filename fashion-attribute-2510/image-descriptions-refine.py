import json
import re
from tqdm import tqdm

# TARGET="women"
TARGET="men"

def remove_repeats(text):
    # Remove consecutive duplicate words
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

with open(f'./generated/image_{TARGET}_bekleidung_descriptions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

descriptions = []
for key, desc in tqdm(data.items()):
    # Remove file extension and split by '_'
    base, color = key.replace('.jpg', '').split('_', 1)
    # Remove repeated words
    cleaned_desc = remove_repeats(desc)
    descriptions.append({
        "baseProductCode": base.lstrip('0'),
        "colorCode": color,
        "description": cleaned_desc
    })

result = {"descriptions": descriptions}

with open(f'./generated/image_{TARGET}_bekleidung_descriptions_refined.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
