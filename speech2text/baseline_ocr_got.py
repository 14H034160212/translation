
import os
import cv2
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# Usage: python baseline_ocr_got.py

VIDEO_DIR = "extracted_data/闪婚幸运草的命中注定"
RESULTS_DIR = "baseline_results/got_ocr"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model():
    print("Loading GOT-OCR2.0...")
    model_name = "stepfun-ai/GOT-OCR2_0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True, pad_token_id=tokenizer.eos_token_id).eval().cuda()
    return model, tokenizer

def process_video(video_path, model, tokenizer, fps=1):
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
            # GOT-OCR often works on cropped or full images
            h, w, _ = frame.shape
            crop = frame[int(h*0.7):h, :, :]
            
            # Save temp image for GOT (it often takes file path or PIL)
            temp_path = "temp_ocr.png"
            cv2.imwrite(temp_path, crop)
            
            # GOT Inference
            # res = model.chat(tokenizer, temp_path, ocr_type='ocr')
            # Actually GOT has a specific API
            with torch.no_grad():
                res = model.chat(tokenizer, temp_path, ocr_type='ocr')
            
            if res.strip():
                results.append({
                    "video_file": video_name,
                    "predicted_text": res.strip(),
                    "timestamp": round(timestamp, 2)
                })
        
        frame_count += 1
    
    cap.release()
    if os.path.exists("temp_ocr.png"): os.remove("temp_ocr.png")
    return results

def main():
    # GOT-OCR might need specific setup
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Failed to load GOT-OCR: {e}")
        return
        
    video_files = sorted(list(Path(VIDEO_DIR).glob("*.mp4")))[:5]
    
    all_results = []
    for video_file in video_files:
        res = process_video(video_file, model, tokenizer)
        all_results.extend(res)
        
    with open(f"{RESULTS_DIR}/detailed_results.json", "w", encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {RESULTS_DIR}/detailed_results.json")

if __name__ == "__main__":
    main()
