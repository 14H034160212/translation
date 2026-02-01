import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from PIL import Image

# Configuration
MODEL_PATH = "OpenGVLab/InternVL2-4B"
DATA_DIR = "extracted_data/闪婚幸运草的命中注定" # Example path
RESULTS_DIR = "baseline_results/internvl2"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

def load_model():
    print("Loading InternVL2-4B...")
    # InternVL2 usually requires trust_remote_code=True
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor

def run_inference(model, processor, image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Simple prompt for OCR
    prompt = "Please recognize the Chinese subtitles in this image."
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)
    
    generation_args = {
        "max_new_tokens": 1024,
        "do_sample": False,
    }
    
    generated_ids = model.generate(**inputs, **generation_args)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def main():
    model, processor = load_model()
    
    # Iterate over video frames (assuming extracted frames exist or we extract them)
    # Reusing extracted frames from previous steps if available
    # Or extract a few for testing
    
    # Just a placeholder loop for demonstration on how we'd evaluate
    print("Model loaded successfully. Ready for evaluation loop.")
    
    # For baseline comparison, we ideally need the same test set as Qwen3-VL
    # We can assume a list of frames exists.
    
if __name__ == "__main__":
    main()
