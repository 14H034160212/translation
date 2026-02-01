import os
import sys
from pathlib import Path
import argparse
import json

# Reuse QwenVLEvaluator logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen3vl_evaluator import Qwen3VLEvaluator

class BaselineEvaluator(Qwen3VLEvaluator):
    def load_prediction_data(self):
        # Base loader
        data = super().load_prediction_data()
        
        # Deduplication (always useful for frame-based OCR)
        processed_data = []
        for video_file in sorted(set(item['video_file'] for item in data)):
            video_preds = [item for item in data if item['video_file'] == video_file]
            video_preds.sort(key=lambda x: x.get('timestamp', 0))
            
            if not video_preds:
                continue
                
            dedup_video = [video_preds[0]]
            for i in range(1, len(video_preds)):
                curr_text = video_preds[i]['predicted_text'].strip()
                last_text = dedup_video[-1]['predicted_text'].strip()
                if curr_text != last_text:
                    dedup_video.append(video_preds[i])
            
            processed_data.extend(dedup_video)
            
        print(f"✂️ Deduplicated from {len(data)} to {len(processed_data)} records.")
        self.raw_data = processed_data
        return self.raw_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., trocr, paddleocr)")
    parser.add_argument("--results_dir", type=str, help="Override results dir")
    args = parser.parse_args()
    
    results_dir = args.results_dir or f"baseline_results/{args.model}"
    
    evaluator = BaselineEvaluator(
        results_dir=results_dir,
        evaluation_mode="text_only"
    )
    evaluator.run_qwenvl_evaluation(output_filename=f"{args.model}_eval_results.json")

if __name__ == "__main__":
    main()
