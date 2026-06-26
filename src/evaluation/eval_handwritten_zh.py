import os
import json
import re
import jiwer
import unicodedata

# --- CONFIGURATION ---
DATASET_TYPE = "handwritten_zh"
MODELS = [
    "baidu_ocr",
    "deepseekOCR",
    "deepseekOCR2",
    "dots_mocr",
    "mineru",
    "monkey_ocr",
    "paddle_vl_1.5",
    "paddle_vl_1.6",
    "tesseract",
]

GT_DIR = os.path.join("data", "raw", DATASET_TYPE, "gt")


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize(text):
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove markdown image tags, e.g. ![alt](path)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Strip LaTeX wrappers, e.g. $$\text{...}$$ or $...$
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", r"\1", text, flags=re.DOTALL)

    # Standardize punctuation
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Remove all spaces
    text = "".join(text.split())

    return text.strip()


# -----------------------------
# EXTRACT PREDICTION HELPERS
# -----------------------------
def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _strip_html(text):
    # Replace tags with spaces so adjacent table cells don't merge together.
    return re.sub(r"<[^>]+>", " ", text)


def _join_content_list(items):
    parts = []
    for item in items:
        for key in ("text", "content", "table_body"):
            val = item.get(key)
            if val:
                parts.append(val)
                break
    return "\n".join(parts)


# -----------------------------
# EXTRACT PREDICTION
# -----------------------------
def extract_pred_text(file_id, model_type, results_dir):
    try:
        sample_dir = os.path.join(results_dir, file_id)

        # -- Baidu Unlimited OCR --
        if model_type == "baidu_ocr":
            return _strip_html(_read_file(os.path.join(sample_dir, "result.md")))

        # -- DeepSeek-OCR / DeepSeek-OCR2 --
        if model_type in ("deepseekOCR", "deepseekOCR2"):
            return _read_file(os.path.join(sample_dir, "result.mmd"))

        # -- Tesseract --
        elif model_type == "tesseract":
            return _read_file(os.path.join(sample_dir, "result.txt"))

        # -- dots.ocr --
        elif model_type == "dots_mocr":
            return _read_file(os.path.join(sample_dir, f"{file_id}.md"))

        # -- PaddleOCR-VL (1.5 / 1.6) --
        elif model_type in ("paddle_vl_1.5", "paddle_vl_1.6"):
            pred_path = os.path.join(sample_dir, f"{file_id}_res.json")
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            res_list = pred_data.get("parsing_res_list", [])
            return "\n".join(b.get("block_content", "") for b in res_list)

        # -- MinerU --
        elif model_type == "mineru":
            pred_path = os.path.join(
                sample_dir, file_id, "vlm", f"{file_id}_content_list.json"
            )
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            return _join_content_list(pred_data)

        # -- MonkeyOCR --
        elif model_type == "monkey_ocr":
            pred_path = os.path.join(
                sample_dir, file_id, f"{file_id}_content_list.json"
            )
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            return _strip_html(_join_content_list(pred_data))

    except Exception as e:
        print(f"[WARNING] {model_type} missing file for {file_id}: {e}")
        return ""


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def run_evaluation():
    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith(".json")]

    print(f"Evaluating {len(gt_files)} Chinese samples...\n")

    for MODEL_NAME in MODELS:
        print("=" * 40)
        print(f"Evaluating model: {MODEL_NAME}")
        print("=" * 40)

        RESULTS_DIR = os.path.join("outputs", DATASET_TYPE, MODEL_NAME)
        REPORT_PATH = os.path.join(
            "evaluation_reports",
            DATASET_TYPE,
            f"{MODEL_NAME}_eval_report.json"
        )

        results = []
        all_gt = []
        all_pred = []

        for i, gt_file in enumerate(gt_files, 1):
            file_id = os.path.splitext(gt_file)[0]

            print(f"[{i}/{len(gt_files)}] {file_id}")

            # --- Load GT ---
            with open(os.path.join(GT_DIR, gt_file), "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_text = normalize(gt_data.get("text", ""))

            # --- Load Prediction ---
            raw_pred_text = extract_pred_text(file_id, MODEL_NAME, RESULTS_DIR)
            pred_text = normalize(raw_pred_text)
            
            # -- Metric --
            cer = jiwer.cer(gt_text, pred_text) if gt_text else 1.0

            results.append({
                "id": file_id,
                "gt": gt_text,
                "pred": pred_text,
                "cer": cer
            })

            all_gt.append(gt_text)
            all_pred.append(pred_text)

        # --- Global Metric ---
        total_cer = jiwer.cer(all_gt, all_pred)

        print("-" * 30)
        print(f"FINAL RESULTS for {MODEL_NAME}")
        print(f"Average CER: {total_cer:.4f} ({total_cer*100:.2f}%)")
        print("-" * 30)

        # --- Save report ---
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_NAME,
                "summary": {
                    "average_cer": total_cer
                },
                "details": results
            }, f, indent=4, ensure_ascii=False)
    
        print(f"Saved: {REPORT_PATH}\n")

if __name__ == "__main__":
    run_evaluation()
