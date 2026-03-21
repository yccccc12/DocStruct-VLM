import os
import glob
import requests
import base64
import json
import time
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
DATASET_TYPES = ["handwritten_en", "handwritten_zh"]  # Change to "handwritten_en" for English dataset
MODEL_NAME = "paddle_vl"                            # Model identifier for output organization

# API Credentials from .env
API_URL = os.getenv("PaddleOCR_VL_API_URL")
TOKEN = os.getenv("PaddleOCR_VL_API_TOKEN")

# --- DYNAMIC PATH SETTINGS ---
# INPUT_FOLDER = os.path.join("data", "raw", DATASET_TYPE, "images") 
# BASE_OUTPUT_DIR = os.path.join("outputs", DATASET_TYPE, MODEL_NAME)

def process_document_with_paddle(file_path, api_url, token, output_dir="output", **options):
    """
    Sends a file to the PaddleOCR-VL API and saves the resulting Markdown and images.
    
    Args:
        file_path (str): Path to the input image or PDF.
        api_url (str): The endpoint URL for the OCR service.
        token (str): Your API authorization token.
        output_dir (str): Directory where results will be saved.
        **options: Optional flags like useDocUnwarping, useChartRecognition, etc.
    """

    doc_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(output_dir, doc_name)
    
    file_extension = os.path.splitext(file_path)[1].lower()
    file_type = 0 if file_extension == ".pdf" else 1
    
    try:
        with open(file_path, "rb") as file:
            file_data = base64.b64encode(file.read()).decode("ascii")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "file": file_data,
        "fileType": file_type,
        "useDocOrientationClassify": options.get("useDocOrientationClassify", False),
        "useDocUnwarping": options.get("useDocUnwarping", False),
        "useChartRecognition": options.get("useChartRecognition", False),
    }

    print(f"Processing {file_path}...")
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"API Error: {response.status_code} - {response.text}")
        return

    result = response.json().get("result", {})
    os.makedirs(output_dir, exist_ok=True)

    # Parse and Save Results
    for i, res in enumerate(result.get("layoutParsingResults", [])):
        # Save Markdown
        md_text = res.get("markdown", {}).get("text", "")
        md_filename = os.path.join(output_dir, f"doc_{i}.md")
        with open(md_filename, "w", encoding="utf-8") as md_file:
            md_file.write(md_text)
        
        # Save Inline Markdown Images
        for img_path, img_url in res.get("markdown", {}).get("images", {}).items():
            _save_image_from_url(img_url, os.path.join(output_dir, img_path))
            
        # Save Layout/Output Images
        for img_name, img_url in res.get("outputImages", {}).items():
            save_path = os.path.join(output_dir, f"{img_name}_{i}.jpg")
            _save_image_from_url(img_url, save_path)

        # Save Pruned Result
        pruned_result = res.get("prunedResult", {})
        with open(os.path.join(output_dir, f"pruned_result_{i}.json"), "w", encoding="utf-8") as json_file:
            json.dump(pruned_result, json_file, indent=4)

    print(f"Processing complete. Files saved in: {output_dir}")

def _save_image_from_url(url, save_path):
    """Helper to download and save images."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_response = requests.get(url)
    if img_response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(img_response.content)
    else:
        print(f"Failed to download image from {url}")

def run_batch_ocr():
    for dataset in DATASET_TYPES:
        input_folder = os.path.join("data", "raw", dataset, "images")
        base_output_dir = os.path.join("outputs", dataset, MODEL_NAME)

        os.makedirs(base_output_dir, exist_ok=True)

        # Start dataset timer
        dataset_start_time = time.time()

        # Find images
        image_extensions = ("*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))

        if not image_files:
            print(f"No images found in: {os.path.abspath(input_folder)}")
            continue

        print(f"\n{'='*40}")
        print(f"DATASET: {dataset}")
        print(f"Found:   {len(image_files)} images")
        print(f"{'='*40}")

        # Process images
        for i, img_path in enumerate(image_files, 1):
            img_name = os.path.basename(img_path)
            print(f"\n[{i}/{len(image_files)}] Processing: {img_name}")

            try:
                process_document_with_paddle(
                    file_path=img_path,
                    api_url=API_URL,
                    token=TOKEN,
                    output_dir=base_output_dir
                )
            except Exception as e:
                print(f"!!! Error processing {img_name}: {e}")

        # End dataset timer
        dataset_elapsed = time.time() - dataset_start_time

        print(f"\n✅ Finished dataset: {dataset}")
        print(f"⏳ Total time: {dataset_elapsed:.2f} seconds")
        print(f"⏳ Avg per image: {dataset_elapsed / len(image_files):.2f} seconds")

    print("\n" + "="*40)
    print("All datasets processed!")
    print("="*40)

if __name__ == "__main__":
    run_batch_ocr()