import os
import json
import time
import torch
import whisper
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import fuzz
import sacrebleu
import jiwer
import re

# Add current dir to path to import fusion logic if needed
import sys
sys.path.append(os.getcwd())

from speech2text.qwenvl_whisper_fusion import QwenVLWhisperFusion, FusionResult

class FusionAblator:
    def __init__(self, ocr_json_path, video_dir, gt_dir, whisper_model="medium"):
        self.ocr_json_path = ocr_json_path
        self.video_dir = Path(video_dir)
        self.gt_dir = Path(gt_dir)
        self.whisper_model_name = whisper_model
        
        # Initialize Fusion System
        # We set threshold to 60 as dummy, we will override it in the ablation loop
        self.fusion_system = QwenVLWhisperFusion(
            whisper_model=whisper_model,
            frame_interval=0.2, # 5fps
            similarity_threshold=60.0
        )
        
        # Load OCR data
        print(f"Loading OCR data from {ocr_json_path}...")
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            self.ocr_raw_data = json.load(f)
            
        # Group OCR by video
        self.ocr_by_video = {}
        for item in self.ocr_raw_data:
            v_file = item['video_file']
            if v_file not in self.ocr_by_video:
                self.ocr_by_video[v_file] = []
            self.ocr_by_video[v_file].append((item['timestamp'], item['predicted_text']))
            
        print(f"Loaded OCR for {len(self.ocr_by_video)} videos.")

    def run_ablation(self, thresholds=[0, 20, 40, 50, 60, 70, 80, 100], max_videos=None):
        video_files = sorted(list(self.video_dir.glob("*.mp4")))
        if max_videos:
            video_files = video_files[:max_videos]
            
        all_threshold_results = []
        
        # Pre-transcribe all videos with Whisper to use cache
        print("Ensuring Whisper transcriptions are cached...")
        for video_path in tqdm(video_files, desc="TTS Transcription"):
            video_name = video_path.stem
            # This will use the internal cache mechanism of QwenVLWhisperFusion
            self.fusion_system.transcribe_with_whisper(str(video_path), video_name)
            
        for threshold in thresholds:
            print(f"\n--- Testing Threshold: {threshold}% ---")
            self.fusion_system.similarity_threshold = threshold
            
            video_results = []
            for video_path in tqdm(video_files, desc=f"Evaluating T={threshold}"):
                video_name = video_path.stem
                video_file = video_path.name
                
                if video_file not in self.ocr_by_video:
                    continue
                    
                # Get OCR segments
                ocr_segments = self.ocr_by_video[video_file]
                
                # Get Whisper segments from cache
                whisper_result = self.fusion_system.transcribe_with_whisper(str(video_path), video_name)
                whisper_segments = whisper_result["segments"]
                whisper_original_text = whisper_result["text"].strip()
                
                # Align and Fuse
                final_text, _ = self.fusion_system.align_and_fuse(ocr_segments, whisper_segments)
                
                # Get Ground Truth
                gt_text = self.fusion_system.get_ground_truth(video_name)
                if not gt_text:
                    continue
                    
                # Calculate metrics
                metrics = self.fusion_system.calculate_metrics(final_text, gt_text)
                
                video_results.append({
                    "video": video_file,
                    "bleu": metrics['bleu_score'],
                    "cer": metrics['cer'],
                    "char_acc": metrics['character_accuracy'],
                    "composite": metrics['composite_score']
                })
            
            if not video_results:
                continue
                
            # Aggregate stats for this threshold
            df = pd.DataFrame(video_results)
            threshold_summary = {
                "threshold": threshold,
                "avg_bleu": df['bleu'].mean(),
                "avg_cer": df['cer'].mean(),
                "avg_char_acc": df['char_acc'].mean(),
                "avg_composite": df['composite'].mean()
            }
            all_threshold_results.append(threshold_summary)
            print(f"Summary for T={threshold}: BLEU={threshold_summary['avg_bleu']:.4f}, Composite={threshold_summary['avg_composite']:.4f}")
            
        # Save results
        results_df = pd.DataFrame(all_threshold_results)
        output_path = "speech2text/fusion_ablation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nAblation study complete. Results saved to {output_path}")
        
        return results_df

if __name__ == "__main__":
    ablator = FusionAblator(
        ocr_json_path="baseline_results/qwen3vl_4b_fps5/qwen3vl_baseline_final/detailed_results.json",
        video_dir="extracted_data/闪婚幸运草的命中注定",
        gt_dir="extracted_data/闪婚幸运草的命中注定/带角色标注的字幕"
    )
    
    # Run ablation on a subset first for quick verification, then full if needed
    # Using a meaningful range of thresholds
    ablator.run_ablation(thresholds=[0, 20, 40, 50, 60, 70, 80, 100], max_videos=20)
