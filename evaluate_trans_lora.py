
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import json
import os
from pathlib import Path
import sacrebleu

# Configuration
BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "translation/chinese_japanese_lora_output/fold_0/final_model"
OUTPUT_DIR = "translation/eval_results_qwen25_lora"
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
    
    # Load validation data from the fold split
    split_path = Path("translation/chinese_japanese_lora_output/fold_0/data_splits.json")
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
        
    val_ids = splits.get("val_ids", [])
    print(f"Evaluating on {len(val_ids)} validation samples...")
    
    # Since we need the text, we reload the data. Ideally we should have saved the text in split info, 
    # but we can reload aligned pairs and filter by ID.
    
    data_dir = Path("extracted_data")
    chinese_dir = data_dir / "Chinese"
    japanese_dir = data_dir / "Japanese"
    
    eval_data = []
    
    for file_id in val_ids:
        c_path = chinese_dir / f"{file_id}.txt"
        j_path = japanese_dir / f"{file_id}.txt"
        
        if c_path.exists() and j_path.exists():
            with open(c_path, 'r') as f: c_text = f.read().strip()
            with open(j_path, 'r') as f: j_text = f.read().strip()
            
            # Use same cleaning as training
            c_text = import_cleaning_func(c_text) 
            j_text = import_cleaning_func(j_text)
            
            eval_data.append({
                "id": file_id,
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
        json.dump([
            {"src": item["chinese"], "ref": item["japanese"], "pred": p}
            for item, p in zip(eval_data, preds)
        ], f, ensure_ascii=False, indent=2)

def import_cleaning_func(text):
    import re
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = normalized.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r'^\s*\d+\s*[：:]\s*', '', line).strip()
        cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines).strip()

if __name__ == "__main__":
    main()
