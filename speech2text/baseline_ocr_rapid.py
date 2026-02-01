
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from rapidocr_onnxruntime import RapidOCR

def process_video(video_path, ocr_engine, fps=1):
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        return []
    frame_interval = max(1, int(video_fps / fps))
    
    results = []
    frame_count = 0
    success = True
    
    video_name = Path(video_path).name
    
    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            
            # Crop bottom 30%
            h, w, _ = frame.shape
            crop = frame[int(h*0.7):h, :, :]
            
            # RapidOCR expects standard numpy image (BGR is fine for simple OCR, or RGB)
            # RapidOCR returns: [[[[x1,y1]..], text, score], ...]
            # Note: RapidOCR API differs slightly from PaddleOCR
            
            result, elapse = ocr_engine(crop)
            
            if result:
                text_lines = []
                for line in result:
                    # line structure: [coords, text, score]
                    text = line[1]
                    score = float(line[2])
                    if score > 0.5:
                         text_lines.append(text)
                
                full_text = " ".join(text_lines).strip()
                if full_text:
                    results.append({
                        "video_file": video_name,
                        "predicted_text": full_text,
                        "timestamp": round(timestamp, 2)
                    })
        
        frame_count += 1
        
    cap.release()
    return results

def main():
    parser = argparse.ArgumentParser(description="RapidOCR (PaddleOCR ONNX) Baseline")
    parser.add_argument("--data_dir", type=str, default="extracted_data/闪婚幸运草的命中注定", help="Path to video directory")
    parser.add_argument("--output_dir", type=str, default="baseline_results/rapidocr", help="Output directory")
    args = parser.parse_args()

    # Initialize RapidOCR
    ocr = RapidOCR()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    video_files = sorted(list(Path(args.data_dir).glob("*.mp4")))
    print(f"Processing {len(video_files)} videos with RapidOCR...")

    cache_file = output_dir / "detailed_results.json"
    processed_files = set()
    
    # Simple cache logic
    if cache_file.exists():
         with open(cache_file, 'r', encoding='utf-8') as f:
             results = json.load(f)
         processed_files = set(r['video_file'] for r in results)
         print(f"Loaded {len(results)} cached records.")

    for video_file in tqdm(video_files):
        if video_file.name in processed_files:
            continue
            
        video_res = process_video(video_file, ocr)
        results.extend(video_res)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(results)} items to {cache_file}")

if __name__ == "__main__":
    main()
