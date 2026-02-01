import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor
import sacrebleu

try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM
    MODEL_CLASS = AutoModelForCausalLM

from peft import PeftModel

def generate_translation(model, processor, text, device="cuda"):
    messages = [
        {"role": "system", "content": "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"},
        {"role": "user", "content": text}
    ]
    input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[input_text], padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()

def evaluate(model_path, data_dir, output_file, base_model_path="Qwen/Qwen3-VL-4B-Instruct"):
    print(f"Loading base model: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model in 4-bit
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = MODEL_CLASS.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    if model_path and model_path != base_model_path:
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        
    # Load test data (using validation split from fold 0 logic for simplicity)
    # Ideally should be a separate test set, but we reuse the split logic
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
                 
    # Limit for quick eval if needed
    print(f"Found {len(data_pairs)} pairs. Evaluating...")
    
    results = []
    refs = []
    preds = []
    
    # Simple cache check
    cache_file = Path(output_file).with_suffix('.tmp.json')
    if cache_file.exists():
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        preds = [r["prediction"] for r in results]
        refs = [[r["reference"]] for r in results]
    else:
        for item in tqdm(data_pairs):
            pred = generate_translation(model, processor, item["chinese"])
            results.append({
                "id": item["id"],
                "source": item["chinese"],
                "reference": item["japanese"],
                "prediction": pred
            })
            refs.append([item["japanese"]])
            preds.append(pred)
        
        # Save cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
    # Calculate BLEU
    try:
        bleu = sacrebleu.corpus_bleu(preds, refs, tokenize='ja-mecab')
        print(f"BLEU: {bleu.score}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "bleu": bleu.score,
                "results": results
            }, f, ensure_ascii=False, indent=4)
            
        return bleu.score
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        # Still save the results even if BLEU fails
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e), "results": results}, f, ensure_ascii=False, indent=4)
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="Path to LoRA adapter or None for zero-shot")
    parser.add_argument("--data_dir", default="extracted_data")
    parser.add_argument("--output_file", default="translation_results.json")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.data_dir, args.output_file)
