
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

# Implementation of v2 training with optimized params
def main():
    # Optimized params for small dataset (79 pairs)
    # 1. Lower LR: 5e-5 (was 2e-4) -> More stable on small data
    # 2. Higher Rank: r=32, alpha=64 (was 16/32) -> More capacity
    # 3. More Epochs: 20 (was 10) -> Let it converge gently
    
    from train_translation_qwen3vl import Qwen3VLTranslationTrainer
    
    trainer = Qwen3VLTranslationTrainer(
        output_dir="translation/qwen3vl_translation_lora_v2_output",
        n_folds=5,
        current_fold=0
    )
    trainer.setup_model_and_processor()
    
    # Custom config for v2
    trainer.setup_lora(r=32, lora_alpha=64, lora_dropout=0.05)
    
    trainer.train(
        num_epochs=20,
        learning_rate=5e-5,
        batch_size=1
    )

if __name__ == "__main__":
    main()
