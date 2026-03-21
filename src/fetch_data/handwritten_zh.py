import requests
import os
import json

# Save data to data/raw/handwritten_zh
save_dir = "data/raw/handwritten_zh"
images_dir = os.path.join(save_dir, "images")
gt_dir = os.path.join(save_dir, "gt")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# Parameters to fetch first 20 rows from the test split
dataset_name = "Teklia/CASIA-HWDB2-line"
split = "test"
offset = 0
length = 20

# URL and parameters
url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": dataset_name,
    "config": "default",
    "split": split,
    "offset": offset,
    "length": length
}

# Fetch data
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    rows = data.get("rows", [])
    
    for i, item in enumerate(rows):

        row_data = item['row']

        
        image_url = row_data['image']['src']
        transcription = row_data['text']

        file_id = f"handwritten_zh_{i + offset:04d}"


        # Save image
        img_resp = requests.get(image_url)
        img_path = os.path.join(images_dir, f"{file_id}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_resp.content)
        
        # Save GT text
        gt_path = os.path.join(gt_dir, f"{file_id}.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump({"text": transcription}, f, ensure_ascii=False, indent=2)

    print(f"Successfully fetched {len(rows)} samples from {dataset_name}.")
else:
    print(f"Error fetching data: {response.status_code} - {response.text}")
