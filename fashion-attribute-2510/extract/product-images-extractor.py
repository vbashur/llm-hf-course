import os
import json
import requests
from tqdm import tqdm

target_category = "men_bekleidung"
target_host = "<HOST_NAME>"
target_folder = f"../input/{target_category}"

json_file_path = f"../input/{target_category}.json"

# Load JSON data from file
with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)


# Ensure target folder exists
os.makedirs(target_folder, exist_ok=True)

# Process each item
for item in tqdm(json_data["hits"]["hits"], desc="Downloading images"):
    source = item["_source"]
    fields = item["fields"]

    base_code = source["baseProductCode"]
    color_code = source["colorCode"]
    image_path = fields["firstLowRes"][0]

    image_url = target_host + image_path
    filename = f"{base_code}_{color_code}.jpg"
    filepath = os.path.join(target_folder, filename)

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        # print(f"Saved image to {filepath}")
    except requests.RequestException as e:
        print(f"Failed to fetch image from {image_url}: {e}")
    # quit(1)