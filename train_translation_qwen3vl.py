#!/usr/bin/env python3
"""
Chinese to Japanese LoRA Fine-tuning for Qwen3-VL
"""

import os
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
import re
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
# Try specific import, fallback to AutoModel
try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM
    MODEL_CLASS = AutoModelForCausalLM

from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)

class Qwen3VLTranslationTrainer:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
                 data_dir: str = "extracted_data",
                 output_dir: str = "translation/qwen3vl_translation_lora_output",
                 use_4bit: bool = True,
                 n_folds: int = 5,
                 current_fold: int = 0):
        
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        self.n_folds = n_folds
        self.current_fold = current_fold
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fold_output_dir = self.output_dir / f"fold_{current_fold}"
        self.fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_config = {
            "model_name": model_name,
            "use_4bit": use_4bit,
            "n_folds": n_folds,
            "current_fold": current_fold,
            "timestamp": datetime.now().isoformat()
        }
        
    def clean_speaker_identifier(self, text: str) -> str:
        """Remove per-line speaker identifiers."""
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = normalized.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r'^\s*\d+\s*[：:]\s*', '', line).strip()
            cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines).strip()
        
    def setup_model_and_processor(self):
        print("🔧 Loading model and processor...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        quantization_config = None
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
        self.model = MODEL_CLASS.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_4bit else "auto"
        )
        
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        print("✅ Model and processor loaded")
        
    def setup_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        self.training_config.update({
            "lora_r": r, "lora_alpha": lora_alpha, "target_modules": target_modules
        })
        
    def load_and_preprocess_data(self) -> DatasetDict:
        print("📊 Loading data...")
        chinese_dir = self.data_dir / "Chinese"
        japanese_dir = self.data_dir / "Japanese"
        
        chinese_files = {}
        japanese_files = {}
        
        for file_path in sorted(chinese_dir.glob("*.txt")):
            with open(file_path, 'r', encoding='utf-8') as f:
                chinese_files[file_path.stem] = f.read().strip()
        for file_path in sorted(japanese_dir.glob("*.txt")):
            with open(file_path, 'r', encoding='utf-8') as f:
                japanese_files[file_path.stem] = f.read().strip()
        
        aligned_data = []
        for file_id in chinese_files:
            if file_id in japanese_files:
                c_clean = self.clean_speaker_identifier(chinese_files[file_id])
                j_clean = self.clean_speaker_identifier(japanese_files[file_id])
                if c_clean and j_clean:
                    aligned_data.append({"id": file_id, "chinese": c_clean, "japanese": j_clean})
                    
        print(f"✅ Found {len(aligned_data)} aligned pairs")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        train_indices, val_indices = list(kf.split(aligned_data))[self.current_fold]
        
        train_data = [aligned_data[i] for i in train_indices]
        val_data = [aligned_data[i] for i in val_indices]
        print(f"📈 Fold {self.current_fold+1}: Train={len(train_data)}, Val={len(val_data)}")
        
        # Format for Chat Template
        def format_example(example):
            messages = [
                {"role": "system", "content": "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"},
                {"role": "user", "content": example["chinese"]},
                {"role": "assistant", "content": example["japanese"]}
            ]
            # Use processor's chat template
            # For training, we normally want the full text. 
            # transformers Trainer usually expects 'labels' to be 'input_ids' for CLM (shifted inside model)
            # We can use apply_chat_template output as text.
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
            
        train_texts = [format_example(item)["text"] for item in train_data]
        val_texts = [format_example(item)["text"] for item in val_data]
        
        return DatasetDict({
            "train": Dataset.from_dict({"text": train_texts}),
            "validation": Dataset.from_dict({"text": val_texts})
        })
        
    def tokenize_function(self, examples):
        # We assume processor(text=...) behaves like tokenizer(text=...) for Qwen3-VL processor
        # Qwen2-VL processor returns input_ids, attention_mask, pixel_values (if any)
        # Since we have no images, pixel_values might be absent, which we confirmed is fine.
        return self.processor(
            text=examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024, # Adjust based on GPU memory
            return_tensors="pt"
        )
        
    def train(self, num_epochs=3, learning_rate=2e-4, batch_size=4):
        dataset_dict = self.load_and_preprocess_data()
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.fold_output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=5,
            save_steps=50,
            eval_steps=50,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=True,
            report_to=["tensorboard"]
        )
        
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        
        final_path = self.fold_output_dir / "final_model"
        trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))
        print(f"✅ Training complete. Model saved to {final_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1) # Qwen3-VL is huge, reduce batch size
    args = parser.parse_args()
    
    trainer = Qwen3VLTranslationTrainer()
    trainer.setup_model_and_processor()
    trainer.setup_lora()
    trainer.train(num_epochs=args.num_epochs, batch_size=args.batch_size)
