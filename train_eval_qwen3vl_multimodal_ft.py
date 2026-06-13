#!/usr/bin/env python3
"""
Qwen3-VL-4B multimodal LoRA fine-tuning pilot (fold 0).

Reviewer issue: the visual-context ablation (+3.35 BLEU) was only run in the
zero-shot setting; the released FT model is text-only. This pilot fine-tunes
with video frames included in BOTH training and evaluation, holding everything
else identical to the text-only v2 recipe (QLoRA 4-bit, r=32 α=64, lr 5e-5,
20 epochs, early-stop patience 3, same alphabetical-sort fold split), so the
comparison against the text-only FT fold-0 result (BLEU 19.68) isolates the
contribution of visual input to fine-tuning.

Frames: 4 per episode, uniformly sampled, long side resized to 640 px.

Run:
  CUDA_VISIBLE_DEVICES=2 venv_chattts/bin/python train_eval_qwen3vl_multimodal_ft.py --fold 0
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import sacrebleu
import torch
from PIL import Image
from sklearn.model_selection import KFold
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
try:
    from transformers import Qwen3VLForConditionalGeneration as ModelCls
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelCls

MODEL_ID  = "Qwen/Qwen3-VL-4B-Instruct"
BASE      = Path(__file__).parent
ZH_DIR    = BASE / "extracted_data/Chinese"
JA_DIR    = BASE / "extracted_data/Japanese"
VIDEO_DIR = BASE / "extracted_data/闪婚幸运草的命中注定"
OUT_DIR   = BASE / "qwen3vl_multimodal_ft_output"
RES_DIR   = BASE / "baseline_results"

SYSTEM_PROMPT = "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"
STRIP_RE   = re.compile(r"\d+:\s*")
NUM_FRAMES = 4
FRAME_LONG_SIDE = 640
N_FOLDS    = 5


def strip_nums(t):
    return STRIP_RE.sub("", t).strip()

def clean_speaker_ids(t):
    lines = re.sub(r"\r\n|\r", "\n", t).split("\n")
    return "\n".join(re.sub(r"^\s*\d+\s*[：:]\s*", "", l).strip() for l in lines).strip()


def load_pairs():
    pairs = []
    for f in sorted(ZH_DIR.glob("*.txt")):          # alphabetical, matches v2 split
        jf = JA_DIR / f.name
        if jf.exists():
            c = clean_speaker_ids(f.read_text("utf-8").strip())
            j = clean_speaker_ids(jf.read_text("utf-8").strip())
            v = VIDEO_DIR / f"{f.stem}.mp4"
            if c and j:
                pairs.append({"id": f.stem, "chinese": c, "japanese": j,
                              "video": v if v.exists() else None})
    n_vid = sum(1 for p in pairs if p["video"])
    print(f"Loaded {len(pairs)} pairs ({n_vid} with video)")
    return pairs


def get_fold_split(pairs, fold):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    tr, va = list(kf.split(pairs))[fold]
    return [pairs[i] for i in tr], [pairs[i] for i in va]


def extract_frames(video_path, num_frames=NUM_FRAMES):
    frames = []
    if video_path is None:
        return frames
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        for idx in np.linspace(0, total - 1, num_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w, h = img.size
                scale = FRAME_LONG_SIDE / max(w, h)
                if scale < 1:
                    img = img.resize((int(w * scale), int(h * scale)))
                frames.append(img)
    cap.release()
    return frames


def build_messages(p, frames, with_answer):
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": p["chinese"]})
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": content},
    ]
    if with_answer:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": p["japanese"]}]})
    return msgs


def encode_sample(p, processor):
    """Return dict with input_ids, attention_mask, labels, pixel_values, image_grid_thw."""
    frames = extract_frames(p["video"])
    full_msgs   = build_messages(p, frames, with_answer=True)
    prompt_msgs = build_messages(p, frames, with_answer=False)

    full = processor.apply_chat_template(
        full_msgs, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=False)
    prompt = processor.apply_chat_template(
        prompt_msgs, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True)

    input_ids = full["input_ids"][0]
    prompt_len = prompt["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    sample = {
        "input_ids":      input_ids,
        "attention_mask": full["attention_mask"][0],
        "labels":         labels,
    }
    for k in ("pixel_values", "image_grid_thw"):
        if k in full:
            sample[k] = full[k]
    return sample


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor
        self.cache = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if i not in self.cache:
            self.cache[i] = encode_sample(self.pairs[i], self.processor)
        return self.cache[i]


def collate(batch):
    # batch_size=1 throughout — pass tensors straight through
    assert len(batch) == 1
    b = batch[0]
    out = {
        "input_ids":      b["input_ids"].unsqueeze(0),
        "attention_mask": b["attention_mask"].unsqueeze(0),
        "labels":         b["labels"].unsqueeze(0),
    }
    if "pixel_values" in b:
        out["pixel_values"]   = b["pixel_values"]
        out["image_grid_thw"] = b["image_grid_thw"]
    return out


def load_model(adapter=None):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = ModelCls.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    return model, processor


def train_fold(fold, pairs):
    train_pairs, val_pairs = get_fold_split(pairs, fold)
    print(f"Fold {fold}: train={len(train_pairs)}, val={len(val_pairs)}")
    fold_dir = OUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = EpisodeDataset(train_pairs, processor)
    val_ds   = EpisodeDataset(val_pairs,   processor)

    args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
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
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    adapter = fold_dir / "final_adapter"
    model.save_pretrained(str(adapter))
    processor.save_pretrained(str(adapter))
    print(f"Adapter saved → {adapter}")
    del model
    torch.cuda.empty_cache()
    return adapter


def eval_fold(fold, pairs, adapter):
    _, val_pairs = get_fold_split(pairs, fold)
    model, processor = load_model(adapter)
    model.eval()

    results = []
    for p in val_pairs:
        frames = extract_frames(p["video"])
        msgs = build_messages(p, frames, with_answer=False)
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4096, do_sample=False,
                                 temperature=None, top_p=None)
        pred = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        results.append({"id": p["id"], "source": p["chinese"],
                        "reference": p["japanese"], "prediction": pred})
        print(f"  [{p['id']}] {strip_nums(pred)[:80]}")

    preds = [strip_nums(r["prediction"]) for r in results]
    refs  = [strip_nums(r["reference"])  for r in results]
    bleu  = sacrebleu.corpus_bleu(preds, [refs], tokenize="ja-mecab").score
    chrf  = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    out = {"fold": fold, "n_val": len(results), "BLEU": round(bleu, 4),
           "chrF++": round(chrf, 4), "num_frames": NUM_FRAMES, "results": results}
    out_path = RES_DIR / f"qwen3vl_multimodal_ft_fold{fold}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nFold {fold} (multimodal FT): BLEU={bleu:.4f}  chrF++={chrf:.4f}  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs()
    adapter = OUT_DIR / f"fold_{args.fold}" / "final_adapter"
    if not args.skip_train:
        adapter = train_fold(args.fold, pairs)
    eval_fold(args.fold, pairs, adapter)


if __name__ == "__main__":
    main()
