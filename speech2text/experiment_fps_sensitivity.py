
import os
import cv2
import easyocr
import json
from pathlib import Path
from tqdm import tqdm

# Usage: python experiment_fps_sensitivity.py

VIDEO_PATH = "extracted_data/闪婚幸运草的命中注定/11.mp4"
RESULTS_DIR = "baseline_results/fps_experiment"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment_at_fps(fps, reader):
    print(f"Testing OCR at {fps} fps...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    detected_texts = []
    frame_count = 0
    success = True
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames // frame_interval)
    
    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            # Crop bottom for faster OCR
            h, w, _ = frame.shape
            crop = frame[int(h*0.7):h, :, :]
            
            # EasyOCR
            res = reader.readtext(crop, detail=0)
            text = " ".join(res).strip()
            if text:
                detected_texts.append(text)
            pbar.update(1)
            
        frame_count += 1
    
    cap.release()
    pbar.close()
    
    # Deduplicate consecutive identical
    deduped = []
    if detected_texts:
        deduped.append(detected_texts[0])
        for i in range(1, len(detected_texts)):
            if detected_texts[i] != deduped[-1]:
                deduped.append(detected_texts[i])
    
    return " ".join(deduped)

def main():
    reader = easyocr.Reader(['ch_sim', 'en'])
    
    # We only take the first minute to save time
    # Actually, 11.mp4 is 81s, let's just do it all.
    
    fps_list = [1, 2, 5]
    final_texts = {}
    
    for fps in fps_list:
        text = run_experiment_at_fps(fps, reader)
        final_texts[f"{fps}fps"] = text
        
    with open(f"{RESULTS_DIR}/comparison.json", "w", encoding='utf-8') as f:
        json.dump(final_texts, f, ensure_ascii=False, indent=2)
        
    print("\nFPS Comparison Complete.")
    for fps, text in final_texts.items():
        print(f"{fps}: {len(text)} chars")

if __name__ == "__main__":
    main()
