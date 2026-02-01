import sys
import os
from pathlib import Path

# Reuse QwenVLEvaluator logic but point to Paddle results
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen3vl_evaluator import Qwen3VLEvaluator

def main():
    evaluator = Qwen3VLEvaluator(
        results_dir="baseline_results/paddleocr",
        evaluation_mode="text_only"
    )
    # This will calculate metrics comparing Paddle output with Ground Truth SRTs
    evaluator.run_qwenvl_evaluation(output_filename="paddle_eval_results.json")

if __name__ == "__main__":
    main()
