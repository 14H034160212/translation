import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
from pathlib import Path

# Usage: python baseline_trans_qwen25.py

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "baseline_results/qwen25_translation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    print("Loading Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def translate_text(model, tokenizer, text):
    prompt = f"Translate the following Chinese subtitle to Japanese:\n{text}"
    messages = [
        {"role": "system", "content": "You are a professional subtitle translator."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Dataset
    data_dir = Path("extracted_data")
    chinese_dir = data_dir / "Chinese"
    japanese_dir = data_dir / "Japanese"
    
    data_pairs = [] # List of dicts
    for f in chinese_dir.glob("*.txt"):
        if (japanese_dir / f.name).exists():
             with open(f, 'r') as cf, open(japanese_dir / f.name, 'r') as jf:
                 data_pairs.append({
                     "id": f.stem,
                     "chinese": cf.read().strip(),
                     "japanese": jf.read().strip()
                 })
    
    print(f"Found {len(data_pairs)} pairs. Evaluating...")
    import sacrebleu

    results = []
    refs = []
    preds = []
    
    count = 0 
    for item in tqdm(data_pairs):
        trans = translate_text(model, tokenizer, item["chinese"])
        
        results.append({
            "id": item["id"],
            "source": item["chinese"],
            "reference": item["japanese"],
            "prediction": trans
        })
        refs.append([item["japanese"]])
        preds.append(trans)
        
        # Save intermediate
        count += 1
        if count % 10 == 0:
            with open(f"{OUTPUT_DIR}/results_partial.json", "w", encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(preds, refs, tokenize='ja-mecab')
    print(f"Qwen2.5-7B-Instruct BLEU: {bleu.score}")
        
    with open(f"{OUTPUT_DIR}/results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
