import os
import torch
import json
import time
from pathlib import Path
from PIL import Image
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from accelerate import Accelerator

def extract_frames(video_path, interval=1.0):
    """Extract frames from video at fixed intervals."""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: return []
    
    interval_frames = int(fps * interval)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval_frames == 0:
            timestamp = count / fps
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(frame_rgb), timestamp))
        count += 1
    cap.release()
    return frames

def run_inference(model_id, video_dir, output_dir, batch_size=8):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("Installing qwen-vl-utils...")
        import subprocess
        subprocess.check_call(["pip", "install", "qwen-vl-utils"])
        from qwen_vl_utils import process_vision_info

    accelerator = Accelerator()
    device = accelerator.device
    
    print(f"Loading model {model_id}...")
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForCausalLM
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto", 
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    # Use a fixed directory for incremental results to allow resuming
    timestamp_str = time.strftime("%Y%m%d") # Group by day
    session_dir = output_dir / f"qwen3vl_session_{timestamp_str}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob("*.mp4"))
    
    # Load processed videos to skip
    processed_files = set()
    for f in session_dir.glob("results_*.json"):
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
                # Ensure it's valid data
                if isinstance(data, list) and len(data) > 0:
                     processed_files.add(f.stem.replace("results_", "") + ".mp4")
        except:
            pass
            
    print(f"Found {len(processed_files)} already processed videos. Skipping them.")

    all_detailed_results = []
    
    # Reload existing results into memory for final merge
    for f in session_dir.glob("results_*.json"):
         with open(f, 'r') as jf:
             all_detailed_results.extend(json.load(jf))

    for video_file in video_files:
        if video_file.name in processed_files:
            continue
            
        print(f"Processing {video_file.name}...")
        frames = extract_frames(video_file)
        
        video_results = []
        
        # Batch processing
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            
            batch_texts = []
            batch_image_inputs = []
            batch_video_inputs = []
            
            for image, timestamp in batch_frames:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Extract all subtitle text from this image. Only output the text, nothing else."},
                        ],
                    }
                ]
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                
                batch_texts.append(text)
                batch_image_inputs.extend(image_inputs) # Flatten list of lists if needed, or append? 
                # process_vision_info returns a list of images for one message. 
                # processor expects 'images' to be a list of all images across all examples? 
                # No, for batching, it usually expects a list of list if it's per sample?
                # Actually, for Qwen2-VL processor:
                # images (List[ImageInput] or List[List[ImageInput]]): 
                # If List[ImageInput], they correspond to the images in the input text in order.
                # Since each text has one image, we can just extend a single list of all images.
                
            inputs = processor(
                text=batch_texts,
                images=batch_image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            for j, output_text in enumerate(output_texts):
                video_results.append({
                    "video_file": str(video_file.name),
                    "frame_index": i + j,
                    "timestamp": batch_frames[j][1],
                    "predicted_text": output_text.strip()
                })
        
        # Incremental Save
        video_result_path = session_dir / f"results_{video_file.name.replace('.mp4', '')}.json"
        with open(video_result_path, 'w', encoding='utf-8') as f:
            json.dump(video_results, f, ensure_ascii=False, indent=4)
            
        all_detailed_results.extend(video_results)
            
    # Save unified results
    result_path = output_dir / f"qwen3vl_baseline_final"
    result_path.mkdir(exist_ok=True)
    
    with open(result_path / "detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_detailed_results, f, ensure_ascii=False, indent=4)
        
    print(f"All results saved to {result_path}")
    return result_path

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct" 
    VIDEO_DIR = "extracted_data/闪婚幸运草的命中注定"
    OUTPUT_DIR = "baseline_results"
    
    run_inference(MODEL_ID, VIDEO_DIR, OUTPUT_DIR, batch_size=8)
