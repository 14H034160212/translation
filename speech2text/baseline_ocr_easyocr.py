import os
import cv2
import easyocr
import json
import sys
from tqdm import tqdm
from pathlib import Path

# Initialize EasyOCR reader
# 'ch_sim' = Simplified Chinese, 'en' = English
reader = easyocr.Reader(['ch_sim', 'en'])

def main():
    video_dir_str = None
    for dirpath, _, files in os.walk("extracted_data"):
        if any(f.endswith('.mp4') for f in files):
            video_dir_str = dirpath
            print(f"Found video dir: {video_dir_str}")
            break
            
    if not video_dir_str or not os.path.exists(video_dir_str):
        print(f"Dir not found via walk: {video_dir_str}")
        return
    
    video_files = [os.path.join(video_dir_str, f) for f in os.listdir(video_dir_str) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos.")

    output_dir = Path("baseline_results/easyocr")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # video_files already populated above
    if not video_files:
        print("No videos found in dir!")
        return

    print(f"Found {len(video_files)} videos. Processing with EasyOCR...")
    
    results = []
    
    for video_path_str in tqdm(video_files):
        video_path = Path(video_path_str)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Process 1 frame per second
        frame_interval = int(fps)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # EasyOCR expects RGB or path
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = reader.readtext(frame_rgb, detail=0)
                
                if result:
                    text = " ".join(result)
                    timestamp = frame_idx / fps
                    results.append({
                        "video_file": video_path.name,
                        "predicted_text": text,
                        "timestamp": timestamp
                    })
            
            frame_idx += 1
        cap.release()

    # Save results
    with open(output_dir / "detailed_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("EasyOCR processing complete.")

if __name__ == "__main__":
    main()
