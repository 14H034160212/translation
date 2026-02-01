
import os
import cv2
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Usage: python baseline_ocr_trocr.py

VIDEO_DIR = "extracted_data/闪婚幸运草的命中注定"
RESULTS_DIR = "baseline_results/trocr"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model():
    print("Loading TrOCR model...")
    # Using printed version for subtitles
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device

def process_video(video_path, model, processor, device, fps=1):
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    results = []
    frame_count = 0
    success = True
    
    video_name = Path(video_path).name
    print(f"Processing {video_name}...")
    
    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            
            # Simple crop for potential subtitle area (bottom 30%)
            h, w, _ = frame.shape
            subtitle_crop = frame[int(h*0.7):h, :, :]
            
            # Convert to PIL
            image = Image.fromarray(cv2.cvtColor(subtitle_crop, cv2.COLOR_BGR2RGB))
            
            # TrOCR Inference
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if generated_text.strip():
                results.append({
                    "video_file": video_name,
                    "predicted_text": generated_text.strip(),
                    "timestamp": round(timestamp, 2)
                })
        
        frame_count += 1
    
    cap.release()
    return results

def main():
    model, processor, device = load_model()
    
    video_files = sorted(list(Path(VIDEO_DIR).glob("*.mp4")))
    # For baseline, we process first 5 videos to be fast
    video_files = video_files[:5]
    
    all_results = []
    for video_file in video_files:
        res = process_video(video_file, model, processor, device)
        all_results.extend(res)
        
    with open(f"{RESULTS_DIR}/detailed_results.json", "w", encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {RESULTS_DIR}/detailed_results.json")

if __name__ == "__main__":
    main()
