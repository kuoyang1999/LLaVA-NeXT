import os
from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset("lmms-lab/LLaVA-OneVision-Data", name="raven(cauldron)", split="all").select(range(5))

data_name = "onevision2M"

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create paths relative to current directory
image_folder = os.path.join(current_dir, data_name)
os.makedirs(image_folder, exist_ok=True)

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.jpg"
        # Create the full directory path for the image
        image_path = os.path.join(image_folder, json_data["image"])
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        da["image"].save(image_path)
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)

output_json = os.path.join(current_dir, f"{data_name}.json")
with open(output_json, "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
