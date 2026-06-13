#!/usr/bin/env python3
"""
Gemma-4-12B LoRA fine-tuning pilot for Chinese→Japanese subtitle translation.

Reviewer issue M1: the strongest zero-shot baseline (Gemma-4-12B, BLEU 26.27)
was never fine-tuned. This pilot LoRA-fine-tunes it on fold 0 with the SAME
data split, prompt format, and training protocol as train_eval_qwen3_8b_lora.py
(text-only, QLoRA 4-bit, r=16 α=32, lr 1e-4, 15 epochs, early stop patience 5)
so the result is directly comparable to the Qwen3-8B FT fold-0 number (20.48).

Run:
  CUDA_VISIBLE_DEVICES=1 venv_chattts/bin/python train_eval_gemma4_12b_lora.py --fold 0
"""

import argparse
import json
import re
from pathlib import Path

import sacrebleu
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "google/gemma-4-12B-it"
BASE     = Path(__file__).parent
ZH_DIR   = BASE / "extracted_data/Chinese"
JA_DIR   = BASE / "extracted_data/Japanese"
OUT_DIR  = BASE / "gemma4_12b_lora_output"
RES_DIR  = BASE / "baseline_results"

SYSTEM_PROMPT = "你是一位专业的中日字幕翻译员。只输出翻译结果，不要添加任何解释或额外内容。"
STRIP_RE      = re.compile(r"\d+:\s*")
MAX_LENGTH    = 1024
N_FOLDS       = 5


def strip_nums(text):
    return STRIP_RE.sub("", text).strip()

def strip_speaker_ids(text):
    lines = re.sub(r"\r\n|\r", "\n", text).split("\n")
    return "\n".join(re.sub(r"^\s*\d+\s*[：:]\s*", "", l).strip() for l in lines).strip()

def load_pairs():
    pairs = []
    for f in sorted(ZH_DIR.glob("*.txt")):
        jf = JA_DIR / f.name
        if jf.exists():
            c = strip_speaker_ids(f.read_text("utf-8").strip())
            j = strip_speaker_ids(jf.read_text("utf-8").strip())
            if c and j:
                pairs.append({"id": f.stem, "chinese": c, "japanese": j})
    print(f"Loaded {len(pairs)} pairs")
    return pairs

def get_fold_split(pairs, fold):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    tr_idx, va_idx = list(kf.split(pairs))[fold]
    return [pairs[i] for i in tr_idx], [pairs[i] for i in va_idx]


def build_msgs(p, with_answer):
    # Gemma chat template: system role folded into the first user turn
    user = SYSTEM_PROMPT + "\n\n" + p["chinese"]
    msgs = [{"role": "user", "content": [{"type": "text", "text": user}]}]
    if with_answer:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": p["japanese"]}]})
    return msgs


def build_tokenized_dataset(pairs, processor):
    tok = processor.tokenizer
    all_ids, all_mask, all_labels = [], [], []
    for p in pairs:
        full_text   = processor.apply_chat_template(build_msgs(p, True),  tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(build_msgs(p, False), tokenize=False, add_generation_prompt=True)
        full_enc   = tok(full_text,   max_length=MAX_LENGTH, truncation=True)
        prompt_enc = tok(prompt_text, max_length=MAX_LENGTH, truncation=True)
        ids = full_enc["input_ids"]
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * prompt_len + ids[prompt_len:]
        all_ids.append(ids)
        all_mask.append(full_enc["attention_mask"])
        all_labels.append(labels[:MAX_LENGTH])
    return Dataset.from_dict({"input_ids": all_ids, "attention_mask": all_mask, "labels": all_labels})


def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    print("Base model loaded.")
    return model, processor


def add_lora(model):
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def train_fold(fold, pairs, model, processor):
    train_pairs, val_pairs = get_fold_split(pairs, fold)
    print(f"Fold {fold}: train={len(train_pairs)}, val={len(val_pairs)}")
    fold_dir = OUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_tokenized_dataset(train_pairs, processor)
    val_ds   = build_tokenized_dataset(val_pairs,   processor)
    collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=15,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=30,
        eval_steps=30,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        save_total_limit=2,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    print(f"Training fold {fold} …")
    trainer.train()
    adapter_dir = fold_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Adapter saved → {adapter_dir}")
    return adapter_dir


def eval_fold(fold, pairs, adapter_dir, processor):
    _, val_pairs = get_fold_split(pairs, fold)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    tok = processor.tokenizer

    results = []
    for p in val_pairs:
        text = processor.apply_chat_template(build_msgs(p, False), tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({"id": p["id"], "source": p["chinese"],
                        "reference": p["japanese"], "prediction": pred})
        print(f"  [{p['id']}] {strip_nums(pred)[:80]}")

    preds = [strip_nums(r["prediction"]) for r in results]
    refs  = [strip_nums(r["reference"])  for r in results]
    bleu  = sacrebleu.corpus_bleu(preds, [refs], tokenize="ja-mecab").score
    chrf  = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    out = {"fold": fold, "n_val": len(results), "BLEU": round(bleu, 4),
           "chrF++": round(chrf, 4), "results": results}
    out_path = RES_DIR / f"gemma4_12b_ft_fold{fold}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nFold {fold}: BLEU={bleu:.4f}  chrF++={chrf:.4f}  → {out_path}")
    del model, base
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs()
    adapter_dir = OUT_DIR / f"fold_{args.fold}" / "final_adapter"

    if not args.skip_train:
        model, processor = load_model_and_processor()
        model = add_lora(model)
        adapter_dir = train_fold(args.fold, pairs, model, processor)
        del model
        torch.cuda.empty_cache()
    else:
        processor = AutoProcessor.from_pretrained(str(adapter_dir))

    eval_fold(args.fold, pairs, adapter_dir, processor)


if __name__ == "__main__":
    main()
