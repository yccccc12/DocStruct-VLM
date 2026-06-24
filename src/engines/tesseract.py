import os
import glob
from pathlib import Path
import pytesseract
from PIL import Image

# --- CONFIGURATION ---
MODEL_NAME = "tesseract"

# Datasets to process: (dataset_type, language)
DATASETS = [
    ("handwritten_en", "eng"),
    ("handwritten_zh", "chi_sim"),
]

def run_tesseract(file_path, output_dir, language):
    """
    Runs Tesseract OCR and saves output as:
    outputs/{dataset}/{model}/{filename}/result.txt
    """

    file_path = Path(file_path)
    output_dir = Path(output_dir)

    sample_folder = output_dir / file_path.stem
    output_txt_path = sample_folder / "result.txt"
    sample_folder.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(file_path)

        text = pytesseract.image_to_string(img, lang=language)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[OK] Saved -> {output_txt_path}")

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")


def run_batch_ocr(dataset_type, language):
    # --- DYNAMIC PATH SETTINGS ---
    input_folder = os.path.join("data", "raw", dataset_type, "images")
    base_output_dir = os.path.join("outputs", dataset_type, MODEL_NAME)

    os.makedirs(base_output_dir, exist_ok=True)

    # Recursive search
    file_extensions = ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.tiff")
    file_list = []

    for ext in file_extensions:
        file_list.extend(glob.glob(os.path.join(input_folder, ext), recursive=True))

    if not file_list:
        print(f"No files found in: {os.path.abspath(input_folder)}")
        return

    print("="*40)
    print(f"STARTING BATCH PROCESS")
    print(f"Dataset: {dataset_type}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Found:   {len(file_list)} files")
    print("="*40)

    for i, file_path in enumerate(file_list, 1):
        file_name = os.path.basename(file_path)
        print(f"\n[{i}/{len(file_list)}] Processing: {file_name}")

        run_tesseract(file_path, base_output_dir, language)

    print("\n" + "="*40)
    print("Batch processing complete!")
    print(f"Results saved in: {os.path.abspath(base_output_dir)}")
    print("="*40)


if __name__ == "__main__":
    for dataset_type, language in DATASETS:
        run_batch_ocr(dataset_type, language)