import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
from pathlib import Path
import sacrebleu

# Usage: python baseline_trans_qwen3b_zeroshot.py

MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "baseline_results/qwen3b_translation_zeroshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    print(f"Loading {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def translate_text(model, tokenizer, text):
    system_msg = "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"
    prompt = text
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    template_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([template_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

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
    
    if not chinese_dir.exists() or not japanese_dir.exists():
         print("Data directory not found.")
         return

    data_pairs = []
    for f in chinese_dir.glob("*.txt"):
        if (japanese_dir / f.name).exists():
             with open(f, 'r', encoding='utf-8') as cf, open(japanese_dir / f.name, 'r', encoding='utf-8') as jf:
                 data_pairs.append({
                     "id": f.stem,
                     "chinese": cf.read().strip(),
                     "japanese": jf.read().strip()
                 })
    
    # Cleaning function (same as training/eval)
    def clean_text(text):
        import re
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = normalized.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r'^\s*\d+\s*[：:]\s*', '', line).strip()
            cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines).strip()

    print(f"Found {len(data_pairs)} pairs. Evaluating first 20 for quick baseline...")
    data_pairs = data_pairs[:20]

    results = []
    refs = []
    preds = []
    
    for item in tqdm(data_pairs):
        source = clean_text(item["chinese"])
        target = clean_text(item["japanese"])
        
        prediction = translate_text(model, tokenizer, source)
        
        results.append({
            "id": item["id"],
            "source": source,
            "reference": target,
            "prediction": prediction
        })
        refs.append(target)
        preds.append(prediction)

    # Calculate BLEU
    # Use ja-mecab for Japanese BLEU calculation
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='ja-mecab')
    print(f"{MODEL_PATH} Zero-shot BLEU: {bleu.score}")
        
    with open(f"{OUTPUT_DIR}/results.json", "w", encoding='utf-8') as f:
        json.dump({
            "overall_bleu": bleu.score,
            "results": results
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
