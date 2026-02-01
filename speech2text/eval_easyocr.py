import os
import sys
from pathlib import Path
import json

# Reuse QwenVLEvaluator logic but point to EasyOCR results
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen3vl_evaluator import Qwen3VLEvaluator

def main():
    evaluator = Qwen3VLEvaluator(
        results_dir="baseline_results/easyocr",
        evaluation_mode="text_only"
    )
    evaluator.run_qwenvl_evaluation(output_filename="easyocr_eval_results.json")

if __name__ == "__main__":
    main()
