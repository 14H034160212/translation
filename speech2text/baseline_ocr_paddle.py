import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
import cv2
import sys
from paddleocr import PaddleOCR

# Usage: python speech2text/baseline_ocr_paddle.py <video_path_or_dir>

OUTPUT_DIR = "baseline_results/paddleocr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_paddle_ocr(image_path):
    # PaddleOCR supports Chinese (ch), English (en), etc.
    # use_angle_cls=True helps with rotated text
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False) 
    
    result = ocr.ocr(str(image_path), cls=True)
    
    # Result structure: [ [ [ [x1,y1], [x2,y2], ...], ("text", confidence) ], ... ]
    extracted_text = []
    if result and result[0]:
        for line in result[0]:
            text_content = line[1][0]
            confidence = line[1][1]
            extracted_text.append(text_content)
            
    return "\n".join(extracted_text)

def main():
    import glob
    from pathlib import Path
    
    # Dataset path
    dataset_dir = Path("extracted_data/闪婚幸运草的命中注定")
    video_files = sorted(list(dataset_dir.glob("*.mp4")))
    
    if not video_files:
        print(f"No videos found in {dataset_dir}")
        return

    print(f"Found {len(video_files)} videos. Processing...")
    
    # Initialize OCR once
    ocr = PaddleOCR(use_angle_cls=True, lang='ch') 
    
    results = []
    
    for video_path in video_files:
        print(f"Processing {video_path.name}...")
        cap = cv2.VideoCapture(str(video_path))
        
        # Extract 1 frame per second? Or just middle frame?
        # Qwen2-VL used 1 frame per second.
        # Let's do 1 frame per second to be comparable.
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 24
        
        frame_interval = int(fps) # 1 sec
        frame_idx = 0
        
        video_text = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Run OCR on this frame
                result = ocr.ocr(frame)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        timestamp = frame_idx / fps
                        results.append({
                            "video_file": video_path.name,
                            "predicted_text": text,
                            "timestamp": timestamp
                        })
            
            frame_idx += 1
            
        cap.release()
        
    print("PaddleOCR processing complete. Saving detailed results...")
    import json
    with open(f"{OUTPUT_DIR}/detailed_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
