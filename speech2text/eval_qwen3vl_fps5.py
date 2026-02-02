import os
import sys
from pathlib import Path
import json

# Reuse Qwen3VLEvaluator logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen3vl_evaluator import Qwen3VLEvaluator

class Qwen3VLFPS5Evaluator(Qwen3VLEvaluator):
    def load_prediction_data(self):
        # Load from base class
        # Note: the base class usually looks for latest dir, but we want to specify it
        data = super().load_prediction_data()
        
        # Deduplicate consecutive identical texts per video
        processed_data = []
        # Get unique videos in order they appear
        video_files = []
        seen_videos = set()
        for item in data:
            if item['video_file'] not in seen_videos:
                video_files.append(item['video_file'])
                seen_videos.add(item['video_file'])
        
        for video_file in video_files:
            video_preds = [item for item in data if item['video_file'] == video_file]
            video_preds.sort(key=lambda x: x.get('timestamp', 0))
            
            if not video_preds:
                continue
                
            dedup_video = [video_preds[0]]
            for i in range(1, len(video_preds)):
                # Simple strip and compare
                curr_text = video_preds[i]['predicted_text'].strip()
                last_text = dedup_video[-1]['predicted_text'].strip()
                # Also ignore empty results during dedup if they are redundant
                if curr_text != last_text:
                    # Optional: if it's empty and the last was also empty (though strip handles it)
                    dedup_video.append(video_preds[i])
            
            processed_data.extend(dedup_video)
            
        print(f"✂️ Deduplicated from {len(data)} to {len(processed_data)} records.")
        self.raw_data = processed_data
        return self.raw_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default="baseline_results/qwen3vl_4b_fps5/qwen3vl_baseline_final")
    parser.add_argument('--mode', choices=['segmented', 'text_only'], default='segmented')
    args = parser.parse_args()

    evaluator = Qwen3VLFPS5Evaluator(
        results_dir=args.results_dir,
        evaluation_mode=args.mode
    )
    evaluator.run_qwenvl_evaluation(output_filename=f"qwen3vl_fps5_eval_results_{args.mode}.json")

if __name__ == "__main__":
    main()
