
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
from pathlib import Path

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "extracted_data"
OUTPUT_DIR = "translation/qwen7b_lora_output"

def load_and_preprocess_data():
    chinese_dir = Path(DATA_DIR) / "Chinese"
    japanese_dir = Path(DATA_DIR) / "Japanese"
    
    data = []
    for c_file in chinese_dir.glob("*.txt"):
        j_file = japanese_dir / c_file.name
        if j_file.exists():
            with open(c_file, 'r', encoding='utf-8') as f:
                c_text = f.read().strip()
            with open(j_file, 'r', encoding='utf-8') as f:
                j_text = f.read().strip()
            
            # Simple cleaning
            import re
            c_text = re.sub(r'^\s*\d+\s*[：:]\s*', '', c_text).strip()
            j_text = re.sub(r'^\s*\d+\s*[：:]\s*', '', j_text).strip()
            
            data.append({"instruction": "将中文翻译成日文", "input": c_text, "output": j_text})
    
    return Dataset.from_list(data)

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_and_preprocess_data()
    
    def tokenize_function(examples):
        system_msg = "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"
        prompts = [f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" for text in examples["input"]]
        targets = [f"{text}<|im_end|>" for text in examples["output"]]
        
        model_inputs = tokenizer(prompts, truncation=True, padding=False)
        labels = tokenizer(targets, truncation=True, padding=False)
        
        for i in range(len(model_inputs["input_ids"])):
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i] + labels["input_ids"][i]
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + labels["attention_mask"][i]
            # Mask the prompt length for labels
            labels_ids = [-100] * len(tokenizer(prompts[i])["input_ids"]) + labels["input_ids"][i]
            model_inputs["labels"] = model_inputs.get("labels", [])
            model_inputs["labels"].append(labels_ids)
            
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_32bit",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print(f"Model saved to {OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    train()
