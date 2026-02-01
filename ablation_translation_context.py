import os
import torch
import json
import cv2
import sacrebleu
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Try loading Qwen3VL, fallback to AutoModel
try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM
    MODEL_CLASS = AutoModelForCausalLM

from qwen_vl_utils import process_vision_info

def extract_frames_evenly(video_path, num_frames=8):
    """Extract N frames evenly spaced from the video."""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return []
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
    cap.release()
    return frames

def generate_translation_with_vision(model, processor, text, video_path, device="cuda"):
    frames = extract_frames_evenly(video_path, num_frames=8)
    
    # Check if frames were extracted
    if not frames:
        # Fallback to text-only if video fails
        messages = [
             {"role": "system", "content": "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本。"},
             {"role": "user", "content": text}
        ]
    else:
        # Construct message with images
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
            
        content.append({"type": "text", "text": f"Evaluate the visual context and translate the following subtitles to Japanese:\n\n{text}"})
        
        messages = [
            {"role": "system", "content": "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本。"},
            {"role": "user", "content": content}
        ]
        
    # Prepare inputs
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()

def run_ablation(data_dir, video_dir, output_file, model_path="Qwen/Qwen3-VL-4B-Instruct"):
    print(f"Loading model: {model_path}")
    
    # Load model (using 4-bit to save memory as before)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = MODEL_CLASS.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    data_dir = Path(data_dir)
    video_dir = Path(video_dir)
    chinese_dir = data_dir / "Chinese"
    japanese_dir = data_dir / "Japanese"
    
    # Limit to a subset for speed in ablation
    SUBSET_SIZE = 20
    
    tasks = []
    for f in list(chinese_dir.glob("*.txt"))[:SUBSET_SIZE]:
        vid_file = video_dir / f"{f.stem}.mp4"
        ref_file = japanese_dir / f.name
        
        if vid_file.exists() and ref_file.exists():
            with open(f, 'r') as cf, open(ref_file, 'r') as jf:
                tasks.append({
                    "id": f.stem,
                    "video": vid_file,
                    "source": cf.read().strip(),
                    "reference": jf.read().strip()
                })
                
    print(f"Running ablation on {len(tasks)} samples...")
    
    results = []
    refs = []
    preds = []
    
    for task in tqdm(tasks):
        pred = generate_translation_with_vision(model, processor, task["source"], task["video"])
        
        results.append({
            "id": task["id"],
            "source": task["source"],
            "reference": task["reference"],
            "prediction_visual": pred
        })
        refs.append([task["reference"]])
        preds.append(pred)
        
    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(preds, refs, tokenize='ja-mecab')
    print(f"Visual Context BLEU: {bleu.score}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_ablation(
        "extracted_data",
        "extracted_data/闪婚幸运草的命中注定",
        "ablation_translation_visual_results.json"
    )
