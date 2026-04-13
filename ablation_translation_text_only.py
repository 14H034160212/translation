import torch
import json
import sacrebleu
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM
    MODEL_CLASS = AutoModelForCausalLM

SYSTEM_PROMPT = "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本。"
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
SUBSET_SIZE = 20


def generate_translation_text_only(model, processor, text):
    device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[input_text],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def run_ablation_text(data_dir, output_file, model_path=MODEL_PATH):
    print(f"Loading model: {model_path}")

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
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    data_dir = Path(data_dir)
    chinese_dir = data_dir / "Chinese"
    japanese_dir = data_dir / "Japanese"

    tasks = []
    for f in sorted(chinese_dir.glob("*.txt"))[:SUBSET_SIZE]:
        ref_file = japanese_dir / f.name
        if ref_file.exists():
            with open(f, "r") as cf, open(ref_file, "r") as jf:
                tasks.append({
                    "id": f.stem,
                    "source": cf.read().strip(),
                    "reference": jf.read().strip(),
                })

    print(f"Running text-only ablation on {len(tasks)} samples...")

    results, refs, preds = [], [], []

    for task in tqdm(tasks):
        pred = generate_translation_text_only(model, processor, task["source"])
        results.append({
            "id": task["id"],
            "source": task["source"],
            "reference": task["reference"],
            "prediction_text_only": pred,
        })
        refs.append(task["reference"])
        preds.append(pred)

    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="ja-mecab")
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    print(f"Text Only BLEU: {bleu.score:.4f}  chrF++: {chrf.score:.4f}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_ablation_text(
        "extracted_data",
        "ablation_translation_text_results.json",
    )
