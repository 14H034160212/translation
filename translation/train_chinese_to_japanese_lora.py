#!/usr/bin/env python3
"""
Chinese to Japanese LoRA Fine-tuning
Fine-tune Qwen2.5-3B-Instruct for Chinese-Japanese translation with 5-fold cross-validation.
"""

import os
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
import random
import re
from sklearn.model_selection import KFold

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)

class ChineseToJapaneseLoraTrainer:
    def __init__(self, 
                 model_name: str = "./Qwen2.5-3B-Instruct",
                 data_dir: str = "data",
                 output_dir: str = "translation/chinese_japanese_lora_output",
                 use_4bit: bool = True,
                 n_folds: int = 5,
                 current_fold: int = 0):
        
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        self.n_folds = n_folds
        self.current_fold = current_fold
        
        self.output_dir.mkdir(exist_ok=True)
        
        self.fold_output_dir = self.output_dir / f"fold_{current_fold}"
        self.fold_output_dir.mkdir(exist_ok=True)
        
        self.training_config = {
            "model_name": model_name,
            "use_4bit": use_4bit,
            "n_folds": n_folds,
            "current_fold": current_fold,
            "timestamp": datetime.now().isoformat()
        }
        
    def clean_speaker_identifier(self, text: str) -> str:
        """Remove per-line speaker identifiers like '1:', '2:' etc."""
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = normalized.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r'^\s*\d+\s*[：:]\s*', '', line).strip()
            cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines).strip()
        
    def setup_model_and_tokenizer(self):
        print("🔧 Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_4bit else "auto"
        )
        
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        print("✅ Model and tokenizer loaded")
        
    def setup_lora(self, 
                  r: int = 16,
                  lora_alpha: int = 32, 
                  lora_dropout: float = 0.1,
                  target_modules: List[str] = None):
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
            
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
            "lora_r": r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "lora_dropout": lora_dropout
        })
        
    def load_and_preprocess_data(self) -> DatasetDict:
        print("📊 Loading and preprocessing data...")
        
        chinese_dir = self.data_dir / "Chinese"
        japanese_dir = self.data_dir / "Japanese"
        
        if not chinese_dir.exists() or not japanese_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {chinese_dir} or {japanese_dir}")
            
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
                chinese_text = chinese_files[file_id]
                japanese_text = japanese_files[file_id]
                
                chinese_clean = self.clean_speaker_identifier(chinese_text)
                japanese_clean = self.clean_speaker_identifier(japanese_text)
                
                if chinese_clean and japanese_clean:
                    aligned_data.append({
                        "id": file_id,
                        "chinese": chinese_clean,
                        "japanese": japanese_clean
                    })
        
        print(f"✅ Found {len(aligned_data)} aligned data pairs")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Get indices for the current fold
        all_indices = list(range(len(aligned_data)))
        fold_indices = list(kf.split(all_indices))
        
        if self.current_fold >= len(fold_indices):
            raise ValueError(f"当前折 {self.current_fold} 超出范围，总共只有 {len(fold_indices)} 折")
        
        train_indices, val_indices = fold_indices[self.current_fold]
        
        # Split data
        train_data = [aligned_data[i] for i in train_indices]
        val_data = [aligned_data[i] for i in val_indices]
        
        print(f"📈 第 {self.current_fold + 1} 折 - 训练集: {len(train_data)} 个样本")
        print(f"📊 第 {self.current_fold + 1} 折 - 验证集: {len(val_data)} 个样本")
        
        # Save data split info
        data_splits_info = {
            "n_folds": self.n_folds,
            "current_fold": self.current_fold,
            "total_samples": len(aligned_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "train_ids": [item["id"] for item in train_data],
            "val_ids": [item["id"] for item in val_data]
        }
        
        splits_path = self.fold_output_dir / "data_splits.json"
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(data_splits_info, f, indent=4, ensure_ascii=False)
        print(f"📁 数据分割信息已保存到: {splits_path}")
        
        # Build training data format
        def create_prompt(chinese: str, japanese: str) -> str:
            # Use the exact same format as the evaluation script
            system_msg = "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"
            user_msg = chinese
            assistant_msg = japanese
            
            # Build the full conversation format
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            return prompt
        
        train_texts = [create_prompt(item["chinese"], item["japanese"]) for item in train_data]
        val_texts = [create_prompt(item["chinese"], item["japanese"]) for item in val_data]
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        return dataset_dict
        
    def tokenize_function(self, examples):
        """Tokenization function."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )
        
    def prepare_dataset(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Prepare the dataset for training."""
        print("🔧 正在准备数据集...")
        
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        return tokenized_datasets
        
    def train(self,
             num_epochs: int = 30,
             learning_rate: float = 2e-4,
             batch_size: int = 4,
             gradient_accumulation_steps: int = 4,
             warmup_steps: int = 100,
             save_steps: int = 100,
             eval_steps: int = 100,
             logging_steps: int = 10):
        """Start training."""
        
        # Load and preprocess data
        dataset_dict = self.load_and_preprocess_data()
        tokenized_datasets = self.prepare_dataset(dataset_dict)
        
        # Set training arguments
        training_args = TrainingArguments(
            output_dir=str(self.fold_output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb and other reporters
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            callbacks=[early_stopping]
        )
        
        # Update training config
        self.training_config.update({
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "logging_steps": logging_steps
        })
        
        # Save training config
        config_path = self.fold_output_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_config, f, indent=4, ensure_ascii=False)
            
        print("🚀 开始训练...")
        print(f"📁 输出目录: {self.fold_output_dir}")
        print(f"⚙️ 训练配置已保存到: {config_path}")
        
        # Start training
        trainer.train()
        
        # Save final model
        final_model_path = self.fold_output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        print(f"✅ 训练完成！最终模型已保存到: {final_model_path}")
        
        # Save training history
        if trainer.state.log_history:
            history_path = self.fold_output_dir / "training_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(trainer.state.log_history, f, indent=4, ensure_ascii=False)
            print(f"📊 训练历史已保存到: {history_path}")
            
        return trainer

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chinese to Japanese LoRA Training with 5-Fold Cross-Validation")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct", help="基础模型路径")
    parser.add_argument("--data_dir", default="data", help="数据目录")
    parser.add_argument("--output_dir", default="translation/chinese_japanese_lora_output", help="输出目录")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="是否使用4bit量化")
    parser.add_argument("--n_folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--current_fold", type=int, default=0, help="当前训练的折数（0-4）")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    # Validate fold arguments
    if args.current_fold < 0 or args.current_fold >= args.n_folds:
        raise ValueError(f"当前折数 {args.current_fold} 必须在 0 到 {args.n_folds - 1} 之间")
    
    # Create trainer
    trainer = ChineseToJapaneseLoraTrainer(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_4bit=args.use_4bit,
        n_folds=args.n_folds,
        current_fold=args.current_fold
    )
    
    # Set up model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Configure LoRA
    trainer.setup_lora(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Start training
    trainer.train(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 
