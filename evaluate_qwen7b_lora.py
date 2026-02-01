
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import json
import os
from pathlib import Path
import sacrebleu

# Configuration
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "translation/qwen7b_lora_output/final_model"
OUTPUT_DIR = "translation/eval_results_qwen7b_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    print("Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("Loading LoRA Adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer

def translate(model, tokenizer, text):
    system_msg = "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main():
    model, tokenizer = load_model()
    
    # We'll use the first 20 samples from extracted_data for consistency with zero-shot evaluation
    data_dir = Path("extracted_data")
    chinese_dir = data_dir / "Chinese"
    japanese_dir = data_dir / "Japanese"
    
    eval_data = []
    
    c_files = sorted(list(chinese_dir.glob("*.txt")))[:20] 
    
    for c_path in c_files:
        j_path = japanese_dir / c_path.name
        
        if c_path.exists() and j_path.exists():
            with open(c_path, 'r', encoding='utf-8') as f: c_text = f.read().strip()
            with open(j_path, 'r', encoding='utf-8') as f: j_text = f.read().strip()
            
            # Use same cleaning
            def clean_text(text):
                import re
                normalized = text.replace('\r\n', '\n').replace('\r', '\n')
                lines = normalized.split('\n')
                cleaned_lines = []
                for line in lines:
                    cleaned_line = re.sub(r'^\s*\d+\s*[：:]\s*', '', line).strip()
                    cleaned_lines.append(cleaned_line)
                return '\n'.join(cleaned_lines).strip()
            
            c_text = clean_text(c_text)
            j_text = clean_text(j_text)
            
            eval_data.append({
                "chinese": c_text,
                "japanese": j_text
            })

    preds = []
    refs = []
    
    for item in tqdm(eval_data):
        pred = translate(model, tokenizer, item["chinese"])
        preds.append(pred)
        refs.append(item["japanese"])
        
    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='ja-mecab')
    print(f"BLEU Score: {bleu.score}")
    
    # Save results
    with open(f"{OUTPUT_DIR}/results.json", "w", encoding='utf-8') as f:
        json.dump({
            "bleu": bleu.score,
            "samples": [
                {"src": item["chinese"], "ref": item["japanese"], "pred": p}
                for item, p in zip(eval_data, preds)
            ]
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
