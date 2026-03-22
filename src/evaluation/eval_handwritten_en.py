import os
import json
import jiwer
import re

# --- CONFIGURATION ---
DATASET_TYPE = "handwritten_en"
MODEL_NAME = "mineru"

# Paths
GT_DIR = os.path.join("data", "raw", DATASET_TYPE, "gt")
RESULTS_DIR = os.path.join("outputs", DATASET_TYPE, MODEL_NAME)
EVALUATION_REPORT_PATH = os.path.join("evaluation_reports", DATASET_TYPE, f"{MODEL_NAME}_eval_report.json")

def normalize(text):
    """Standardize text to make comparison fair."""
    if not text: return ""
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split()).strip()

def extract_pred_text(file_id, model_type): 

    # -- PaddleOCR-VL -- 
    # (JSON output)
    if model_type == "paddle_vl":
        pred_path = os.path.join(RESULTS_DIR, file_id, "pruned_result_0.json")

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
            res_list = pred_data.get("parsing_res_list", [])
            return res_list[0].get("block_content", "") if res_list else ""
        
    # -- MonkeyOCR -- 
    # (Markdown output)
    if model_type == "monkey_ocr":
        pred_path = os.path.join(RESULTS_DIR, file_id, f"{file_id}_text_result.md")

        with open(pred_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            return md_content.strip() if md_content else ""
    
    # -- MinerU --
    # (JSON output)
    if model_type == "mineru":
        
        # Example path: outputs/handwritten_en/mineru/handwritten_en_000/ocr/handwritten_en_000_content_list.json
        pred_path = os.path.join(RESULTS_DIR, file_id, "ocr", f"{file_id}_content_list.json")

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        if not pred_data:
            return ""

        item = pred_data[0]
        if item.get("type") == "text":
            return item.get("text", "")
        else:
            return ""
  
# def extract_pred_text_from_json(json_data, model_type):
#     if model_type == "paddle_vl":
#         res_list = json_data.get("parsing_res_list", [])
#         return res_list[0].get("block_content", "") if res_list else ""
    
#     if model_type == "deepseek_ocr":
#         return json_data.get("text", "")
    
        
def run_evaluation():
    # 1. Find all GT files
    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith(".json")]
    
    results = []
    all_gt = []
    all_pred = []

    print(f"Evaluating {len(gt_files)} samples...\n")

    for gt_file in gt_files:
        file_id = os.path.splitext(gt_file)[0] # e.g., "handwritten_en_000"
        
        # Load Ground Truth
        with open(os.path.join(GT_DIR, gt_file), "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_text = normalize(gt_data.get("text", ""))
        
        raw_pred_text = extract_pred_text(file_id, MODEL_NAME)

        # Load Prediction
        # Structure: results/handwritten_en/paddle_vl/handwritten_en_000/pruned_result_0.json
        # pred_path = os.path.join(RESULTS_DIR, file_id, "pruned_result_0.json")
        
        # if not os.path.exists(pred_path):
        #     print(f"Skipping {file_id}: Prediction file not found at {pred_path}")
        #     continue
            

        # with open(pred_path, "r", encoding="utf-8") as f:
        #     pred_data = json.load(f)

        #     raw_pred_text = extract_pred_text_from_json(pred_data, MODEL_NAME)

            # if parsing_list:
            #     raw_pred_text = parsing_list[0].get("block_content", "")
            # else:
            #     raw_pred_text = ""
            
        pred_text = normalize(raw_pred_text)

        # Calculate individual CER and WER for this file
        cer = jiwer.cer(gt_text, pred_text) if gt_text else 1.0
        wer = jiwer.wer(gt_text, pred_text) if gt_text else 1.0

        results.append({
            "id": file_id,
            "gt": gt_text,
            "pred": pred_text,
            "cer": cer,
            "wer": wer
        })
        
        all_gt.append(gt_text)
        all_pred.append(pred_text)

    # 2. Calculate Global Metrics
    total_cer = jiwer.cer(all_gt, all_pred)
    total_wer = jiwer.wer(all_gt, all_pred)

    # 3. Print Summary
    print("-" * 30)
    print(f"FINAL RESULTS for {MODEL_NAME}")
    print(f"Average CER: {total_cer:.4f} ({total_cer*100:.2f}%)")
    print(f"Average WER: {total_wer:.4f} ({total_wer*100:.2f}%)")
    print("-" * 30)

    # 4. Save detailed report
    
    os.makedirs(os.path.dirname(EVALUATION_REPORT_PATH), exist_ok=True)
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"average_cer": total_cer, "average_wer": total_wer},
            "details": results
        }, f, indent=4)

    print(f"Detailed report saved to: {EVALUATION_REPORT_PATH}")

if __name__ == "__main__":
    run_evaluation()