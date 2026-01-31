#!/usr/bin/env python3
"""Run 5-fold cross-validation training."""

import subprocess
import sys
import os
from pathlib import Path

def run_fold_training(fold: int, **kwargs):
    print(f"🚀 Starting training for fold {fold + 1}...")
    
    cmd = [
        sys.executable, "translation/train_chinese_to_japanese_lora.py",
        "--current_fold", str(fold),
        "--n_folds", "5"
    ]
    
    for key, value in kwargs.items():
        if key in ["model_name", "data_dir", "output_dir", "num_epochs", "learning_rate", "batch_size", "lora_r", "lora_alpha"]:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Training completed for fold {fold + 1}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for fold {fold + 1}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🎯 Starting 5-fold cross-validation training")
    
    training_params = {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "data_dir": "data",
        "output_dir": "translation/chinese_japanese_lora_output",
        "num_epochs": 10,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32
    }
    
    success_count = 0
    for fold in range(5):
        if run_fold_training(fold, **training_params):
            success_count += 1
        else:
            print(f"⚠️ Fold {fold + 1} training failed, continuing to next fold")
    
    print(f"\n🎉 5-fold cross-validation training complete!")
    print(f"Successfully completed: {success_count}/5 folds")
    
    if success_count == 5:
        print("✅ All folds trained successfully!")
    else:
        print(f"⚠️ {5 - success_count} fold(s) failed, please check logs")

if __name__ == "__main__":
    main() 