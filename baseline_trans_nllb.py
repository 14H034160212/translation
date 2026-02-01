import os
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def evaluate_nllb(data_dir, output_file, model_id="facebook/nllb-200-distilled-600M"):
    print(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    
    data_dir = Path(data_dir)
    chinese_dir = data_dir / "Chinese"
    japanese_dir = data_dir / "Japanese"
    
    data_pairs = []
    for f in chinese_dir.glob("*.txt"):
        if (japanese_dir / f.name).exists():
             with open(f, 'r') as cf, open(japanese_dir / f.name, 'r') as jf:
                 data_pairs.append({
                     "id": f.stem,
                     "chinese": cf.read().strip(),
                     "japanese": jf.read().strip()
                 })
                 
    print(f"Found {len(data_pairs)} pairs. Evaluating...")
    
    results = []
    refs = []
    preds = []
    
    # NLLB Language Codes: Chinese (zho_Hans) -> Japanese (jpn_Jpan)
    # Checking specific codes for NLLB
    src_lang = "zho_Hans" 
    tgt_lang = "jpn_Jpan"
    
    for item in tqdm(data_pairs):
        # Translate
        inputs = tokenizer(item["chinese"], return_tensors="pt").to(device)
        
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), 
            max_length=128
        )
        pred = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        results.append({
            "id": item["id"],
            "source": item["chinese"],
            "reference": item["japanese"],
            "prediction": pred
        })
        refs.append([item["japanese"]])
        preds.append(pred)
        
    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(preds, refs, tokenize='ja-mecab')
    print(f"NLLB-200 BLEU: {bleu.score}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate_nllb("extracted_data", "baseline_results/nllb_results.json")
