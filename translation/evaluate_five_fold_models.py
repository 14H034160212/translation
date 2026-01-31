#!/usr/bin/env python3
"""
Five-Fold Cross-Validation Model Evaluation Script
Evaluate all five-fold cross-validation models and generate a comprehensive report.
Use serial model loading to avoid running out of GPU memory.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import time
import csv
from datetime import datetime
import torch
import re
import numpy as np
import pandas as pd
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unified_translator_evaluator import TranslationEvaluator

class FiveFoldModelEvaluator:
    def __init__(self, 
                 base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 output_dir: str = "translation/chinese_japanese_lora_output",
                 test_data_path: str = None):
        
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        
        # If no test data path is provided, reconstruct from split info
        if test_data_path:
            self.test_data_path = Path(test_data_path)
        else:
            self.test_data_path = None
            
        # Helper to strip speaker identifiers
        self.clean_speaker_identifier = lambda text: re.sub(r'^\d+:\s*', '', text.strip())
        
        # Store evaluation results only (no models)
        self.fold_results = {}
        
    def load_base_model(self):
        """Load the base model for comparison."""
        print("🔧 正在加载基础模型...")
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 基础模型加载完成")
        
    def unload_base_model(self):
        """Unload the base model to free GPU memory."""
        print("🗑️ 正在卸载基础模型...")
        if hasattr(self, 'base_model'):
            del self.base_model
        if hasattr(self, 'base_tokenizer'):
            del self.base_tokenizer
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        print("✅ 基础模型已卸载")
        
    def load_single_fold_model(self, fold: int) -> Tuple[PeftModel, AutoTokenizer]:
        """Load a single fold model."""
        print(f"🔧 正在加载第 {fold + 1} 折模型...")
        
        fold_dir = self.output_dir / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"第 {fold + 1} 折目录不存在: {fold_dir}")
            
        model_path = fold_dir / "final_model"
        if not model_path.exists():
            raise FileNotFoundError(f"第 {fold + 1} 折模型不存在: {model_path}")
            
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA weights
            fold_model = PeftModel.from_pretrained(base_model, str(model_path))
            
            # Load tokenizer
            fold_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if fold_tokenizer.pad_token is None:
                fold_tokenizer.pad_token = fold_tokenizer.eos_token
            
            print(f"✅ 第 {fold + 1} 折模型加载完成")
            return fold_model, fold_tokenizer
            
        except Exception as e:
            print(f"❌ 第 {fold + 1} 折模型加载失败: {e}")
            raise
            
    def unload_fold_model(self, model: PeftModel, tokenizer: AutoTokenizer):
        """Unload a fold model to free GPU memory."""
        print("🗑️ 正在卸载折模型...")
        del model
        del tokenizer
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        print("✅ 折模型已卸载")
        
    def load_test_data(self) -> List[Dict]:
        """Load test data."""
        print("📊 正在加载测试数据...")
        
        # If no test data path is provided, reconstruct from split info
        if self.test_data_path is None or not self.test_data_path.exists():
            print("🔍 尝试从数据分割信息重建测试数据...")
            test_data = self.reconstruct_test_data()
        else:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                
        print(f"✅ 加载了 {len(test_data)} 个测试样本")
        return test_data
        
    def reconstruct_test_data(self) -> List[Dict]:
        """Reconstruct test data from split info."""
        print("🔧 从数据分割信息重建测试数据...")
        
        # Load the first fold's split info to get the full dataset
        first_fold_dir = self.output_dir / "fold_0"
        if not first_fold_dir.exists():
            raise FileNotFoundError(f"第一个折目录不存在: {first_fold_dir}")
            
        splits_path = first_fold_dir / "data_splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"数据分割信息不存在: {splits_path}")
            
        with open(splits_path, 'r', encoding='utf-8') as f:
            splits_info = json.load(f)
            
        # Load raw data from the data directory
        data_dir = Path("data")
        chinese_dir = data_dir / "Chinese"
        japanese_dir = data_dir / "Japanese"
        
        if not chinese_dir.exists() or not japanese_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {chinese_dir} 或 {japanese_dir}")
            
        # Read all files
        chinese_files = {}
        japanese_files = {}
        
        for file_path in sorted(chinese_dir.glob("*.txt")):
            with open(file_path, 'r', encoding='utf-8') as f:
                chinese_files[file_path.stem] = f.read().strip()
                
        for file_path in sorted(japanese_dir.glob("*.txt")):
            with open(file_path, 'r', encoding='utf-8') as f:
                japanese_files[file_path.stem] = f.read().strip()
        
        # Rebuild test data (use the first fold's validation set)
        test_data = []
        val_ids = splits_info.get("val_ids", [])
        
        for file_id in val_ids:
            if file_id in chinese_files and file_id in japanese_files:
                chinese_text = chinese_files[file_id]
                japanese_text = japanese_files[file_id]
                
                # Strip speaker identifiers
                chinese_clean = self.clean_speaker_identifier(chinese_text)
                japanese_clean = self.clean_speaker_identifier(japanese_text)
                
                if chinese_clean and japanese_clean:
                    test_data.append({
                        "id": file_id,
                        "chinese": chinese_clean,
                        "japanese": japanese_clean
                    })
        
        print(f"✅ 重建了 {len(test_data)} 个测试样本")
        return test_data
        
    def translate_with_model(self, text: str, model, tokenizer, model_name: str) -> str:
        """Translate using the specified model."""
        try:
            # Strip speaker identifiers
            cleaned_text = self.clean_speaker_identifier(text)
            
            # Build messages in the exact format used by the original script
            messages = [
                {"role": "system", "content": "你是一位专业的翻译员。请将下面的中文文本翻译成日文。只输出翻译后的日文文本，不要添加任何解释或额外内容。"},
                {"role": "user", "content": cleaned_text}
            ]
            
            # Use the same parameters as the original script
            max_tokens = 2048
            temperature = 0.0
            
            # Build inputs
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation (remove input portion)
            translation = response.replace(input_text, "").strip()
            
            return translation
            
        except Exception as e:
            print(f"    ❌ {model_name} 翻译失败: {e}")
            return f"FAILED_{model_name}"
            
    def evaluate_base_model(self, test_data: List[Dict], evaluator: TranslationEvaluator) -> Dict:
        """Evaluate the base model."""
        print("\n🔍 评估基础模型...")
        
        base_hypotheses = []
        base_references = []
        base_comet_scores = []
        
        for i, item in enumerate(test_data):
            print(f"  📝 处理样本 {i+1}/{len(test_data)} (ID: {item['id']})")
            
            chinese_text = item["chinese"]
            japanese_reference = item["japanese"]
            
            translation = self.translate_with_model(
                chinese_text, self.base_model, self.base_tokenizer, "基础模型"
            )
            
            base_hypotheses.append(translation)
            base_references.append(japanese_reference)
            
            # Compute COMET score
            if evaluator.comet_model and "FAILED" not in translation:
                try:
                    comet_data = [{"src": chinese_text, "mt": translation, "ref": japanese_reference}]
                    score = evaluator.comet_model.predict(comet_data, batch_size=1, gpus=0).scores[0]
                    base_comet_scores.append(score)
                except Exception as e:
                    print(f"    ❌ COMET 计算失败: {e}")
        
        # Compute base model metrics
        base_metrics = evaluator.calculate_bleu_chrf_ter(base_hypotheses, base_references)
        if base_comet_scores:
            base_metrics["COMET (Average)"] = sum(base_comet_scores) / len(base_comet_scores)
        else:
            base_metrics["COMET (Average)"] = 0.0
            
        # Save base model results
        self.base_results = {
            "metrics": base_metrics,
            "hypotheses": base_hypotheses,
            "references": base_references,
            "comet_scores": base_comet_scores,
            "test_samples": len(test_data)
        }
        
        print("✅ 基础模型评估完成")
        return base_metrics
        
    def evaluate_single_fold(self, fold: int, test_data: List[Dict], evaluator: TranslationEvaluator) -> Dict:
        """Evaluate a single fold model."""
        print(f"\n🔍 开始评估第 {fold + 1} 折...")
        
        # Load model
        model, tokenizer = self.load_single_fold_model(fold)
        
        try:
            hypotheses = []
            references = []
            sources = []
            comet_scores = []
            
            # Evaluate each test sample
            for i, item in enumerate(test_data):
                print(f"  📝 处理样本 {i+1}/{len(test_data)} (ID: {item['id']})")
                
                chinese_text = item["chinese"]
                japanese_reference = item["japanese"]
                
                # Translate
                translation = self.translate_with_model(
                    chinese_text, model, tokenizer, f"第{fold + 1}折"
                )
                
                hypotheses.append(translation)
                references.append(japanese_reference)
                sources.append(chinese_text)
                
                # Compute COMET score
                if evaluator.comet_model and "FAILED" not in translation:
                    try:
                        comet_data = [{"src": chinese_text, "mt": translation, "ref": japanese_reference}]
                        score = evaluator.comet_model.predict(comet_data, batch_size=1, gpus=0).scores[0]
                        comet_scores.append(score)
                    except Exception as e:
                        print(f"    ❌ COMET 计算失败: {e}")
            
            # Compute evaluation metrics
            metrics = evaluator.calculate_bleu_chrf_ter(hypotheses, references)
            if comet_scores:
                metrics["COMET (Average)"] = sum(comet_scores) / len(comet_scores)
            else:
                metrics["COMET (Average)"] = 0.0
                
            # Save detailed results
            fold_results = {
                "metrics": metrics,
                "hypotheses": hypotheses,
                "references": references,
                "sources": sources,
                "comet_scores": comet_scores,
                "test_samples": len(test_data)
            }
            
            self.fold_results[f"fold_{fold}"] = fold_results
            
            print(f"✅ 第 {fold + 1} 折评估完成")
            return metrics
            
        finally:
            # Ensure model is unloaded
            self.unload_fold_model(model, tokenizer)
        
    def evaluate_all_models(self, output_dir: str = "translation/five_fold_evaluation_results"):
        """Evaluate all models in serial."""
        print("🚀 开始五折交叉验证模型评估（串行模式）")
        print("=" * 80)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        test_data = self.load_test_data()
        
        # Initialize evaluator
        evaluator = TranslationEvaluator()
        
        # Load base model
        self.load_base_model()
        
        # Evaluate base model
        base_metrics = self.evaluate_base_model(test_data, evaluator)
        
        # Unload base model to free GPU memory
        self.unload_base_model()
        
        # Evaluate each fold serially
        print(f"\n🔍 串行评估所有折的模型...")
        fold_metrics = {}
        
        for fold in range(5):
            try:
                metrics = self.evaluate_single_fold(fold, test_data, evaluator)
                fold_metrics[f"fold_{fold}"] = metrics
            except Exception as e:
                print(f"❌ 第 {fold + 1} 折评估失败: {e}")
                continue
        
        # Generate evaluation report
        self.generate_evaluation_report(output_path, base_metrics, fold_metrics, test_data)
        
        print("\n🎉 五折交叉验证评估完成！")
        
    def generate_evaluation_report(self, output_path: Path, base_metrics: Dict, fold_metrics: Dict, test_data: List[Dict]):
        """Generate the evaluation report."""
        print("\n📝 生成评估报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Generate detailed CSV results
        csv_file_path = output_path / f"five_fold_evaluation_{timestamp}.csv"
        self.generate_detailed_csv(csv_file_path, test_data)
        
        # 2. Generate metric comparison report
        summary_file = output_path / f"five_fold_evaluation_summary_{timestamp}.json"
        self.generate_summary_json(summary_file, base_metrics, fold_metrics)
        
        # 3. Print comparison results
        self.print_comparison_results(base_metrics, fold_metrics)
        
        # 4. Generate statistical analysis
        stats_file = output_path / f"five_fold_statistics_{timestamp}.txt"
        self.generate_statistics_report(stats_file, base_metrics, fold_metrics)
        
        print(f"✅ 详细结果已保存到: {csv_file_path}")
        print(f"✅ 对比摘要已保存到: {summary_file}")
        print(f"✅ 统计分析已保存到: {stats_file}")
        
    def generate_detailed_csv(self, csv_file_path: Path, test_data: List[Dict]):
        """Generate the detailed CSV results file."""
        csv_header = ["id", "chinese_text", "japanese_reference", "base_translation"]
        
        # Add columns for each fold
        for fold_name in self.fold_results.keys():
            csv_header.extend([f"{fold_name}_translation", f"{fold_name}_comet"])
            
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()
            
            for i, item in enumerate(test_data):
                row = {
                    "id": item["id"],
                    "chinese_text": item["chinese"],
                    "japanese_reference": item["japanese"]
                }
                
                # Add base model results
                if hasattr(self, 'base_results'):
                    row["base_translation"] = self.base_results["hypotheses"][i]
                
                # Add results for each fold
                for fold_name in self.fold_results.keys():
                    if fold_name in self.fold_results:
                        fold_data = self.fold_results[fold_name]
                        row[f"{fold_name}_translation"] = fold_data["hypotheses"][i]
                        row[f"{fold_name}_comet"] = fold_data["comet_scores"][i] if i < len(fold_data["comet_scores"]) else "N/A"
                
                writer.writerow(row)
                
    def generate_summary_json(self, summary_file: Path, base_metrics: Dict, fold_metrics: Dict):
        """Generate the JSON summary report."""
        summary = {
            "base_model": base_metrics,
            "fold_models": fold_metrics,
            "evaluation_info": {
                "total_folds": len(fold_metrics),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
    def print_comparison_results(self, base_metrics: Dict, fold_metrics: Dict):
        """Print comparison results."""
        print("\n" + "=" * 100)
        print("🎯 五折交叉验证模型评估结果对比")
        print("=" * 100)
        
        # Table header
        header = f"{'指标':<15} {'基础模型':<15}"
        for fold_name in fold_metrics.keys():
            header += f" {fold_name:<15}"
        header += f" {'平均':<15} {'标准差':<15}"
        print(header)
        print("-" * 100)
        
        # Compute stats for each metric
        metrics_list = ["BLEU", "chrF++", "TER", "COMET (Average)"]
        
        for metric in metrics_list:
            if metric in base_metrics:
                base_score = base_metrics[metric]
                row = f"{metric:<15} {base_score:<15.4f}"
                
                fold_scores = []
                for fold_name in fold_metrics.keys():
                    if metric in fold_metrics[fold_name]:
                        score = fold_metrics[fold_name][metric]
                        fold_scores.append(score)
                        row += f" {score:<15.4f}"
                    else:
                        row += f" {'N/A':<15}"
                
                # Compute statistics
                if fold_scores:
                    mean_score = np.mean(fold_scores)
                    std_score = np.std(fold_scores)
                    row += f" {mean_score:<15.4f} {std_score:<15.4f}"
                else:
                    row += f" {'N/A':<15} {'N/A':<15}"
                
                print(row)
                
        print("-" * 100)
        
    def generate_statistics_report(self, stats_file: Path, base_metrics: Dict, fold_metrics: Dict):
        """Generate the statistical analysis report."""
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("五折交叉验证模型评估统计分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 基础模型性能:\n")
            for metric, value in base_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
            f.write("\n")
            
            f.write("📈 各折模型性能:\n")
            for fold_name, metrics in fold_metrics.items():
                f.write(f"  {fold_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.6f}\n")
                f.write("\n")
            
            # Compute improvement stats
            f.write("📊 改进统计:\n")
            metrics_list = ["BLEU", "chrF++", "TER", "COMET (Average)"]
            
            for metric in metrics_list:
                if metric in base_metrics:
                    base_score = base_metrics[metric]
                    fold_scores = []
                    
                    for fold_name in fold_metrics.keys():
                        if metric in fold_metrics[fold_name]:
                            fold_scores.append(fold_metrics[fold_name][metric])
                    
                    if fold_scores:
                        improvements = [fold_score - base_score for fold_score in fold_scores]
                        mean_improvement = np.mean(improvements)
                        std_improvement = np.std(improvements)
                        
                        f.write(f"  {metric}:\n")
                        f.write(f"    平均改进: {mean_improvement:+.6f}\n")
                        f.write(f"    改进标准差: {std_improvement:.6f}\n")
                        f.write(f"    改进范围: {min(improvements):+.6f} ~ {max(improvements):+.6f}\n")
                        f.write(f"    改进样本数: {len(improvements)}/{len(fold_metrics)}\n\n")

def main():
    """Main entry point."""
    print("🚀 开始五折交叉验证模型评估（串行模式）")
    print("=" * 60)
    
    # Create evaluator
    evaluator = FiveFoldModelEvaluator()
    
    # Run evaluation (handles model loading/unloading)
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main() 
