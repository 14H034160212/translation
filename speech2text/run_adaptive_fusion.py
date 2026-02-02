import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from speech2text.qwenvl_whisper_fusion import QwenVLWhisperFusion

def run_adaptive_comparison(max_videos=20):
    ocr_json_path = "baseline_results/qwen3vl_4b_fps5/qwen3vl_baseline_final/detailed_results.json"
    video_dir = Path("extracted_data/闪婚幸运草的命中注定")
    
    # Initialize Fusion System
    fusion_system = QwenVLWhisperFusion(
        whisper_model="medium",
        frame_interval=0.2
    )
    
    # Load OCR data
    with open(ocr_json_path, 'r', encoding='utf-8') as f:
        ocr_raw_data = json.load(f)
        
    ocr_by_video = {}
    for item in ocr_raw_data:
        v_file = item['video_file']
        if v_file not in ocr_by_video:
            ocr_by_video[v_file] = []
        ocr_by_video[v_file].append((item['timestamp'], item['predicted_text']))
        
    video_files = sorted(list(video_dir.glob("*.mp4")))[:max_videos]
    
    results = []
    for video_path in tqdm(video_files, desc="Benchmarking"):
        video_name = video_path.stem
        video_file = video_path.name
        
        if video_file not in ocr_by_video:
            continue
            
        ocr_segments = ocr_by_video[video_file]
        whisper_result = fusion_system.transcribe_with_whisper(str(video_path), video_name)
        whisper_segments = whisper_result["segments"]
        gt_text = fusion_system.get_ground_truth(video_name)
        
        if not gt_text:
            continue
            
        # 1. Static 60% Baseline
        fusion_system.similarity_threshold = 60.0
        static_text, _ = fusion_system.align_and_fuse(ocr_segments, whisper_segments)
        static_metrics = fusion_system.calculate_metrics(static_text, gt_text)
        
        # 2. Adaptive Result
        adaptive_text, _ = fusion_system.align_and_fuse_adaptive(ocr_segments, whisper_segments)
        adaptive_metrics = fusion_system.calculate_metrics(adaptive_text, gt_text)
        
        results.append({
            "video": video_file,
            "static_bleu": static_metrics['bleu_score'],
            "static_composite": static_metrics['composite_score'],
            "adaptive_bleu": adaptive_metrics['bleu_score'],
            "adaptive_composite": adaptive_metrics['composite_score'],
        })
        
    df = pd.DataFrame(results)
    print("\n--- Final Comparison ---")
    print(f"Static 60%  - Avg BLEU: {df['static_bleu'].mean():.4f}, Avg Composite: {df['static_composite'].mean():.4f}")
    print(f"Adaptive T  - Avg BLEU: {df['adaptive_bleu'].mean():.4f}, Avg Composite: {df['adaptive_composite'].mean():.4f}")
    
    diff = df['adaptive_composite'].mean() - df['static_composite'].mean()
    print(f"Gain: {diff:+.4f}")
    
    df.to_csv("speech2text/adaptive_fusion_comparison.csv", index=False)

if __name__ == "__main__":
    run_adaptive_comparison()
