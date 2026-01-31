#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwenvl_evaluator import QwenVLEvaluator

class Qwen3VLEvaluator(QwenVLEvaluator):
    """Qwen3-VL-4B video subtitle recognition evaluator."""
    
    def find_latest_results_dir(self) -> str:
        """Find the latest Qwen3-VL results directory."""
        import glob
        pattern = "baseline_results/qwen3vl_baseline_*"
        result_dirs = glob.glob(pattern)
        if result_dirs:
            latest_dir = max(result_dirs, key=os.path.getmtime)
            return latest_dir
        
        raise FileNotFoundError("未找到任何 Qwen3-VL 结果目录")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Qwen3-VL 统一评估框架')
    parser.add_argument('--mode', choices=['segmented', 'text_only'], default='text_only')
    parser.add_argument('--results-dir', type=str)
    args = parser.parse_args()
    
    evaluator = Qwen3VLEvaluator(
        results_dir=args.results_dir,
        evaluation_mode=args.mode
    )
    evaluator.run_qwenvl_evaluation(output_filename=f"qwen3vl_unified_results_{args.mode}.json")
