#!/usr/bin/env python3
"""Analyze 5-fold cross-validation results."""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_fold_results(output_dir: str):
    output_path = Path(output_dir)
    results = {}
    
    for fold in range(5):
        fold_dir = output_path / f"fold_{fold}"
        if not fold_dir.exists():
            print(f"⚠️ Fold {fold + 1} directory not found: {fold_dir}")
            continue
            
        config_path = fold_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                results[f"fold_{fold}"] = {
                    "config": json.load(f)
                }
        
        splits_path = fold_dir / "data_splits.json"
        if splits_path.exists():
            with open(splits_path, 'r', encoding='utf-8') as f:
                results[f"fold_{fold}"]["splits"] = json.load(f)
        
        history_path = fold_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                results[f"fold_{fold}"]["history"] = json.load(f)
    
    return results

def analyze_training_history(results):
    print("📊 Analyzing training history...")
    
    all_metrics = []
    for fold_name, fold_data in results.items():
        if "history" in fold_data:
            history = fold_data["history"]
            for step_data in history:
                if "eval_loss" in step_data:
                    all_metrics.append({
                        "fold": fold_name,
                        "step": step_data.get("step", 0),
                        "eval_loss": step_data["eval_loss"],
                        "train_loss": step_data.get("loss", None)
                    })
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        best_losses = df.groupby("fold")["eval_loss"].min()
        print("\n🏆 Best validation loss per fold:")
        for fold, loss in best_losses.items():
            print(f"  {fold}: {loss:.4f}")
        
        print(f"\n📈 Overall statistics:")
        print(f"  Average best validation loss: {best_losses.mean():.4f}")
        print(f"  Std dev of best validation loss: {best_losses.std():.4f}")
        print(f"  Range: {best_losses.min():.4f} - {best_losses.max():.4f}")
        
        return df, best_losses
    else:
        print("❌ No training history data found")
        return None, None

def analyze_data_splits(results):
    """Analyze data splits."""
    print("\n📋 分析数据分割...")
    
    splits_info = []
    for fold_name, fold_data in results.items():
        if "splits" in fold_data:
            splits = fold_data["splits"]
            splits_info.append({
                "fold": fold_name,
                "total_samples": splits["total_samples"],
                "train_samples": splits["train_samples"],
                "val_samples": splits["val_samples"],
                "train_ratio": splits["train_samples"] / splits["total_samples"],
                "val_ratio": splits["val_samples"] / splits["total_samples"]
            })
    
    if splits_info:
        df = pd.DataFrame(splits_info)
        print("\n📊 数据分割统计:")
        print(df.to_string(index=False))
        
        # Check data split consistency
        total_samples = df["total_samples"].iloc[0]
        train_samples_mean = df["train_samples"].mean()
        val_samples_mean = df["val_samples"].mean()
        
        print(f"\n🔍 数据分割一致性检查:")
        print(f"  总样本数: {total_samples}")
        print(f"  平均训练样本数: {train_samples_mean:.1f}")
        print(f"  平均验证样本数: {val_samples_mean:.1f}")
        print(f"  训练集比例: {train_samples_mean/total_samples:.3f}")
        print(f"  验证集比例: {val_samples_mean/total_samples:.3f}")
        
        return df
    else:
        print("❌ 没有找到数据分割信息")
        return None

def create_visualizations(results, output_dir: str):
    """Create visualizations."""
    print("\n🎨 创建可视化图表...")
    
    output_path = Path(output_dir)
    
    # Load training history data
    all_metrics = []
    for fold_name, fold_data in results.items():
        if "history" in fold_data:
            history = fold_data["history"]
            for step_data in history:
                if "eval_loss" in step_data:
                    all_metrics.append({
                        "fold": fold_name,
                        "step": step_data.get("step", 0),
                        "eval_loss": step_data["eval_loss"],
                        "train_loss": step_data.get("loss", None)
                    })
    
    if not all_metrics:
        print("❌ 没有足够的数据创建可视化")
        return
    
    df = pd.DataFrame(all_metrics)
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("五折交叉验证训练结果分析", fontsize=16)
    
    # 1. Validation loss curve
    ax1 = axes[0, 0]
    for fold in df["fold"].unique():
        fold_data = df[df["fold"] == fold]
        ax1.plot(fold_data["step"], fold_data["eval_loss"], label=fold, marker='o', markersize=3)
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("验证损失")
    ax1.set_title("验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training loss curve
    ax2 = axes[0, 1]
    for fold in df["fold"].unique():
        fold_data = df[df["fold"] == fold]
        if "train_loss" in fold_data.columns and fold_data["train_loss"].notna().any():
            ax2.plot(fold_data["step"], fold_data["train_loss"], label=fold, marker='s', markersize=3)
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("训练损失")
    ax2.set_title("训练损失曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Best validation loss per fold
    ax3 = axes[1, 0]
    best_losses = df.groupby("fold")["eval_loss"].min()
    folds = list(best_losses.index)
    losses = list(best_losses.values)
    bars = ax3.bar(folds, losses, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    ax3.set_xlabel("折数")
    ax3.set_ylabel("最佳验证损失")
    ax3.set_title("每个折的最佳验证损失")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{loss:.4f}', ha='center', va='bottom')
    
    # 4. Loss distribution box plot
    ax4 = axes[1, 1]
    loss_data = [df[df["fold"] == fold]["eval_loss"].values for fold in folds]
    box_plot = ax4.boxplot(loss_data, labels=folds, patch_artist=True)
    ax4.set_xlabel("折数")
    ax4.set_ylabel("验证损失")
    ax4.set_title("验证损失分布")
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Set colors
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / "five_fold_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 可视化图表已保存到: {plot_path}")
    
    plt.show()

def generate_summary_report(results, output_dir: str):
    """Generate the summary report."""
    print("\n📝 生成汇总报告...")
    
    output_path = Path(output_dir)
    report_path = output_path / "five_fold_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("五折交叉验证训练结果汇总报告\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        f.write("📋 基本信息:\n")
        f.write(f"  输出目录: {output_dir}\n")
        f.write(f"  总折数: 5\n")
        f.write(f"  完成折数: {len(results)}\n\n")
        
        # Per-fold configuration
        f.write("⚙️ 训练配置:\n")
        for fold_name, fold_data in results.items():
            if "config" in fold_data:
                config = fold_data["config"]
                f.write(f"  {fold_name}:\n")
                f.write(f"    模型: {config.get('model_name', 'N/A')}\n")
                f.write(f"    学习率: {config.get('learning_rate', 'N/A')}\n")
                f.write(f"    批次大小: {config.get('batch_size', 'N/A')}\n")
                f.write(f"    训练轮数: {config.get('num_epochs', 'N/A')}\n")
                f.write(f"    LoRA r: {config.get('lora_r', 'N/A')}\n")
                f.write(f"    LoRA alpha: {config.get('lora_alpha', 'N/A')}\n\n")
        
        # Data split info
        f.write("📊 数据分割信息:\n")
        for fold_name, fold_data in results.items():
            if "splits" in fold_data:
                splits = fold_data["splits"]
                f.write(f"  {fold_name}:\n")
                f.write(f"    总样本数: {splits.get('total_samples', 'N/A')}\n")
                f.write(f"    训练样本数: {splits.get('train_samples', 'N/A')}\n")
                f.write(f"    验证样本数: {splits.get('val_samples', 'N/A')}\n\n")
        
        # Training results
        f.write("🏆 训练结果:\n")
        all_metrics = []
        for fold_name, fold_data in results.items():
            if "history" in fold_data:
                history = fold_data["history"]
                eval_losses = [step.get("eval_loss") for step in history if "eval_loss" in step]
                if eval_losses:
                    best_loss = min(eval_losses)
                    final_loss = eval_losses[-1]
                    all_metrics.append((fold_name, best_loss, final_loss))
                    f.write(f"  {fold_name}:\n")
                    f.write(f"    最佳验证损失: {best_loss:.6f}\n")
                    f.write(f"    最终验证损失: {final_loss:.6f}\n\n")
        
        # Overall statistics
        if all_metrics:
            best_losses = [metrics[1] for metrics in all_metrics]
            final_losses = [metrics[2] for metrics in all_metrics]
            
            f.write("📈 总体统计:\n")
            f.write(f"  平均最佳验证损失: {np.mean(best_losses):.6f}\n")
            f.write(f"  最佳验证损失标准差: {np.std(best_losses):.6f}\n")
            f.write(f"  最佳验证损失范围: {min(best_losses):.6f} - {max(best_losses):.6f}\n")
            f.write(f"  平均最终验证损失: {np.mean(final_losses):.6f}\n")
            f.write(f"  最终验证损失标准差: {np.std(final_losses):.6f}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("报告生成完成\n")
    
    print(f"📝 汇总报告已保存到: {report_path}")

def main():
    """Main entry point."""
    output_dir = "translation/chinese_japanese_lora_output"
    
    print("🔍 开始分析五折交叉验证结果...")
    
    # Load results
    results = load_fold_results(output_dir)
    
    if not results:
        print("❌ 没有找到任何训练结果")
        return
    
    print(f"✅ 找到 {len(results)} 折的训练结果")
    
    # Analyze training history
    history_df, best_losses = analyze_training_history(results)
    
    # Analyze data splits
    splits_df = analyze_data_splits(results)
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main() 
