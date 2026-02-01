
import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# Configuration
MODEL_PATH = "OpenGVLab/InternVL2-4B"
RESULTS_DIR = "baseline_results/internvl2"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=1): # Set max_num=1 for speed
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model():
    print("Loading InternVL2-4B...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer

def process_video_frames(model, tokenizer, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_interval = int(fps) 
    frame_idx = 0
    predictions = []
    
    print(f"Processing {video_path.name}...")
    
    # Limit to first 10 seconds for ultra-fast baseline if needed, but let's try 1 frame/sec for full video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            pixel_values = load_image(image, max_num=1).to(torch.bfloat16).to(model.device)
            
            question = "请识别图中的字幕文本。"
            generation_config = dict(
                num_beams=1,
                max_new_tokens=128,
                do_sample=False,
            )
            
            with torch.no_grad():
                response = model.chat(tokenizer, pixel_values, question, generation_config)
            
            predictions.append({
                "video_file": video_path.name,
                "predicted_text": response.strip(),
                "timestamp": frame_idx / fps
            })
            
            # For quick baseline, we only need a few samples per video?
            # No, let's process the full video but only 1 video.
            
        frame_idx += 1
        
    cap.release()
    return predictions

def main():
    video_dir_str = "extracted_data/闪婚幸运草的命中注定"
    if not os.path.exists(video_dir_str):
        print(f"Directory {video_dir_str} not found.")
        return

    # Just 1 representative video for fast result (that has an SRT match)
    video_files = [Path(video_dir_str) / "11.mp4"]
    print(f"Processing single video for fast baseline: {video_files[0]}")

    model, tokenizer = load_model()
    
    all_results = []
    
    for video_path in video_files:
        try:
            results = process_video_frames(model, tokenizer, video_path)
            all_results.extend(results)
            
            with open(Path(RESULTS_DIR) / "detailed_results.json", "w", encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print("InternVL2 (Fast Baseline) processing complete.")

if __name__ == "__main__":
    main()
