
import os
import sys
from pathlib import Path
import json

# Reuse EasyOCR evaluator logic (deduplication) as it is also frame-based
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_easyocr import EasyOCREvaluator

class PaddleOCREvaluator(EasyOCREvaluator):
    pass
    # No overrides needed; logic is identical to EasyOCR (frame-based, needs dedup)

def main():
    evaluator = PaddleOCREvaluator(
        results_dir="baseline_results/paddleocr",
        evaluation_mode="text_only"
    )
    # Output to paddleocr_eval_results_dedup.json
    evaluator.run_qwenvl_evaluation(output_filename="paddleocr_eval_results_dedup.json")

if __name__ == "__main__":
    main()
