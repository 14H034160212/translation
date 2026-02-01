
import os
import sys
from pathlib import Path
import json

# Reuse EasyOCR evaluator logic (deduplication)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_easyocr import EasyOCREvaluator

class RapidOCREvaluator(EasyOCREvaluator):
    pass

def main():
    evaluator = RapidOCREvaluator(
        results_dir="baseline_results/rapidocr",
        evaluation_mode="text_only"
    )
    evaluator.run_qwenvl_evaluation(output_filename="rapidocr_eval_results_dedup.json")

if __name__ == "__main__":
    main()
