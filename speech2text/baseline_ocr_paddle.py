
import os
import cv2
import json
from pathlib import Path
from tqdm import tqdm
try:
    import paddle.inference
    if not hasattr(paddle.inference.Config, 'set_optimization_level'):
        # Fix for PaddlePaddle 3.0+ compatibility with PaddleX
        paddle.inference.Config.set_optimization_level = lambda self, level: None
        print("🔧 Patched paddle.inference.Config.set_optimization_level")
except Exception as e:
    print(f"⚠️ Failed to patch paddle: {e}")

from paddleocr import PaddleOCR

# Usage: python baseline_ocr_paddle.py

VIDEO_DIR = "extracted_data/闪婚幸运草的命中注定"
RESULTS_DIR = "baseline_results/paddleocr"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model():
    print("Loading PaddleOCR...")
    # lang='ch' for Chinese + English
    # Removed use_gpu as it caused issues in 3.4.0
    ocr = PaddleOCR(lang='ch')
    return ocr

def process_video(video_path, ocr, fps=1):
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    results = []
    frame_count = 0
    success = True
    
    video_name = Path(video_path).name
    print(f"Processing {video_name}...")
    
    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            
            # Crop bottom for subtitles
            h, w, _ = frame.shape
            crop = frame[int(h*0.7):h, :, :]
            
            # PaddleOCR Inference
            res = ocr.ocr(crop, cls=True)
            
            texts = []
            if res and res[0]:
                for line in res[0]:
                    texts.append(line[1][0])
            
            combined_text = " ".join(texts).strip()
            
            if combined_text:
                results.append({
                    "video_file": video_name,
                    "predicted_text": combined_text,
                    "timestamp": round(timestamp, 2)
                })
        
        frame_count += 1
    
    cap.release()
    return results

def main():
    ocr = load_model()
    
    video_files = sorted(list(Path(VIDEO_DIR).glob("*.mp4")))
    # Limit to 5 videos for baseline check
    video_files = video_files[:5]
    
    all_results = []
    for video_file in video_files:
        res = process_video(video_file, ocr)
        all_results.extend(res)
        
    with open(f"{RESULTS_DIR}/detailed_results.json", "w", encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {RESULTS_DIR}/detailed_results.json")

if __name__ == "__main__":
    main()
