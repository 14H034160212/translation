#!/usr/bin/env python3
"""
Unified Evaluation Framework
Provides consistent evaluation interface and metrics calculation for QwenVL and Whisper systems.
"""

import re
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import jiwer
import sacrebleu


class EvaluationMetrics:
    """Unified metrics calculator ensuring all systems use consistent methods."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        return text
        # """Standardized text cleaning method."""
        # if not text:
        #     return ""

        # # Remove subtitle prefix
        # text = re.sub(r'^字幕[：:]\s*', '', text)
        
        # # Handle empty content
        # if text.strip() in ['无', '无字幕', '字幕: 无', '字幕：无']:
        #     return ""

        # # Remove anomalous output patterns (prompt content, etc.)
        # anomaly_patterns = [
        #     "这是一个视频截图，请识别其中的中文字幕内容。只返回字幕文字，忽略其他图像元素。",
        #     "这是一个视频截图，请识别其中的中文字幕内容",
        #     "请识别其中的中文字幕内容",
        #     "只返回字幕文字，忽略其他图像元素",
        #     "识别图片中的字幕文字"
        # ]
        # text_stripped = text.strip()
        # for pattern in anomaly_patterns:
        #     if pattern in text_stripped:
        #         return ""

        # # Check for too many prompt keywords
        # prompt_keywords = ['视频截图', '识别其中', '字幕内容', '图像元素', '只返回', '忽略其他']
        # keyword_count = sum(1 for keyword in prompt_keywords if keyword in text_stripped)
        # if keyword_count >= 3:
        #     return ""

        # # Normalize whitespace
        # return re.sub(r'\s+', ' ', text.strip())

    @staticmethod
    def calculate_cer(predicted: str, reference: str) -> float:
        """Calculate Character Error Rate (CER) using jiwer."""
        pred_clean = EvaluationMetrics.clean_text(predicted)
        ref_clean = EvaluationMetrics.clean_text(reference)

        if not ref_clean:
            return 0.0 if not pred_clean else 1.0
        if not pred_clean:
            return 1.0

        # Compute CER at the character level
        return jiwer.cer(ref_clean, pred_clean)

    @staticmethod
    def calculate_character_accuracy(predicted: str, reference: str) -> float:
        """Calculate character accuracy (correct chars / total reference chars)."""
        pred_clean = EvaluationMetrics.clean_text(predicted)
        ref_clean = EvaluationMetrics.clean_text(reference)
        if not ref_clean:
            return 1.0 if not pred_clean else 0.0
        
        if not pred_clean:
            return 0.0

        import jiwer
        
        ref_chars = list(ref_clean)
        pred_chars = list(pred_clean)
        
        # Compute detailed character-level metrics
        output = jiwer.process_characters(ref_clean, pred_clean)
        measures = { 'hits': output.hits, 'substitutions': output.substitutions, 'deletions': output.deletions }

        # 4. Extract core data required for calculation
        # measures['hits'] is the number of correct characters
        correct_chars = measures['hits']
        
        # Total reference characters N = hits (H) + substitutions (S) + deletions (D)
        substitutions = measures['substitutions']
        deletions = measures['deletions']
        total_chars_in_ref = correct_chars + substitutions + deletions

        # 5. Compute final accuracy
        # Re-check denominator to avoid division by zero after cleaning
        if total_chars_in_ref == 0:
            # Reference text was cleaned to empty; safeguard against divide-by-zero
            return 1.0 if not pred_clean else 0.0

        accuracy = correct_chars / total_chars_in_ref
        
        return accuracy

    @staticmethod
    def calculate_bleu_score(predicted: str, reference: str) -> float:
        """Calculate BLEU score using sacrebleu."""
        pred_clean = EvaluationMetrics.clean_text(predicted)
        ref_clean = EvaluationMetrics.clean_text(reference)

        if not ref_clean or not pred_clean:
            return 0.0

        bleu = sacrebleu.corpus_bleu([pred_clean], [[ref_clean]], tokenize='zh')
        return bleu.score / 100.0

    @staticmethod
    def calculate_chrf_plus_plus(predicted: str, reference: str) -> float:
        """Calculate chrF++ score using sacrebleu."""
        pred_clean = EvaluationMetrics.clean_text(predicted)
        ref_clean = EvaluationMetrics.clean_text(reference)

        if not ref_clean or not pred_clean:
            return 0.0

        # Use standard chrF++ parameters (char_order=6, word_order=2, beta=2)
        chrf = sacrebleu.corpus_chrf([pred_clean], [[ref_clean]], word_order=2)
        return chrf.score / 100.0

    @staticmethod
    def calculate_all_metrics(predicted: str, reference: str) -> Dict[str, float]:
        """Calculate all four metrics."""
        return {
            'cer': EvaluationMetrics.calculate_cer(predicted, reference),
            'character_accuracy': EvaluationMetrics.calculate_character_accuracy(predicted, reference),
            'bleu_score': EvaluationMetrics.calculate_bleu_score(predicted, reference),
            'chrf_plus_plus': EvaluationMetrics.calculate_chrf_plus_plus(predicted, reference)
        }


class BaseEvaluator(ABC):
    """Abstract base class for the unified evaluation framework."""
    
    def __init__(self, evaluation_mode: str = "text_only", show_matched_text: bool = True):
        """
        Initialize the evaluator.

        Args:
            evaluation_mode: Evaluation mode
                - "segmented": segment-level matching
                - "text_only": full-text matching
            show_matched_text: Whether to print matched text in the terminal.
        """
        self.evaluation_mode = evaluation_mode
        self.show_matched_text = show_matched_text
        self.metrics_calculator = EvaluationMetrics()
        
        print(f"🔧 评估模式: {evaluation_mode}")
        if show_matched_text:
            print(f"📝 将显示匹配的文本内容")

    @abstractmethod
    def load_reference_data(self) -> Dict[str, Any]:
        """Load reference data (subtitle files, ground truth, etc.)."""
        pass

    @abstractmethod
    def load_prediction_data(self) -> List[Dict]:
        """Load prediction results data."""
        pass

    @abstractmethod
    def match_predictions_to_references(self) -> List[Dict]:
        """Match predictions to reference data."""
        pass

    @abstractmethod
    def get_full_text_for_item(self, item_id: str) -> Tuple[str, str]:
        """Get full predicted and reference text for a specific item."""
        pass

    def calculate_metrics_for_item_segmented(self, matches: List[Dict]) -> Dict:
        """Calculate metrics for a single item (segmented mode)."""
        if not matches:
            return {
                'total_segments': 0,
                'matched_segments': 0,
                'cer': 0.0,
                'character_accuracy': 0.0,
                'bleu_score': 0.0,
                'chrf_plus_plus': 0.0
            }

        # Calculate metrics for each match
        all_metrics = []
        for match in matches:
            predicted = match.get('prediction', '')
            reference = match.get('reference', '')
            metrics = self.metrics_calculator.calculate_all_metrics(predicted, reference)
            all_metrics.append(metrics)

        # Compute averages
        result = {
            'matched_segments': len(matches),
            'cer': sum(m['cer'] for m in all_metrics) / len(all_metrics),
            'character_accuracy': sum(m['character_accuracy'] for m in all_metrics) / len(all_metrics),
            'bleu_score': sum(m['bleu_score'] for m in all_metrics) / len(all_metrics),
            'chrf_plus_plus': sum(m['chrf_plus_plus'] for m in all_metrics) / len(all_metrics)
        }
        
        return result

    def calculate_metrics_for_item_text_only(self, item_id: str) -> Dict:
        """Calculate metrics for a single item (full-text mode)."""
        predicted_text, reference_text = self.get_full_text_for_item(item_id)

        if self.show_matched_text:
            print(f"\n📝 项目 {item_id} 的匹配文本 (整体合并):")
            print(f"参考文本 ({len(reference_text)} 字符): {reference_text}")
            print(f"预测文本 ({len(predicted_text)} 字符): {predicted_text}")
            print("-" * 60)

        metrics = self.metrics_calculator.calculate_all_metrics(predicted_text, reference_text)
        
        return {
            'item_id': item_id,
            'predicted_text_length': len(predicted_text),
            'reference_text_length': len(reference_text),
            'predicted_text': predicted_text,
            'reference_text': reference_text,
            **metrics
        }

    def calculate_overall_metrics(self, item_metrics: List[Dict]) -> Dict:
        """Calculate overall metrics."""
        if not item_metrics:
            return {
                'cer': 1.0,  # Worst CER
                'character_accuracy': 0.0,  # Worst accuracy
                'bleu_score': 0.0,  # Worst BLEU
                'chrf_plus_plus': 0.0,  # Worst chrF++
                'evaluation_mode': self.evaluation_mode,
                'total_items': 0
            }

        if self.evaluation_mode == "text_only":
            # For full-text mode, recompute after concatenation
            all_predicted = " ".join(item['predicted_text'] for item in item_metrics)
            all_reference = " ".join(item['reference_text'] for item in item_metrics)
            overall_metrics = self.metrics_calculator.calculate_all_metrics(all_predicted, all_reference)
        else:
            # For segmented mode, average the metrics
            overall_metrics = {
                'cer': sum(item['cer'] for item in item_metrics) / len(item_metrics),
                'character_accuracy': sum(item['character_accuracy'] for item in item_metrics) / len(item_metrics),
                'bleu_score': sum(item['bleu_score'] for item in item_metrics) / len(item_metrics),
                'chrf_plus_plus': sum(item['chrf_plus_plus'] for item in item_metrics) / len(item_metrics)
            }

        overall_metrics.update({
            'evaluation_mode': self.evaluation_mode,
            'total_items': len(item_metrics)
        })

        return overall_metrics

    def display_results(self, overall_metrics: Dict, item_metrics: List[Dict]):
        """Display evaluation results."""
        mode_name = "整体文本匹配模式" if self.evaluation_mode == "text_only" else "逐句匹配模式"
        print(f"\n📈 评估结果 ({mode_name}):")
        print("=" * 80)

        print(f"\n🎯 总体指标:")
        print(f"  CER (字符错误率): {overall_metrics['cer']:.4f}")
        print(f"  字符准确率: {overall_metrics['character_accuracy']:.4f} ({overall_metrics['character_accuracy']*100:.2f}%)")
        print(f"  BLEU分数: {overall_metrics['bleu_score']:.4f}")
        print(f"  chrF++分数: {overall_metrics['chrf_plus_plus']:.4f}")

        print(f"\n💡 评估说明:")
        print(f"  - 使用统一评估框架 (BaseEvaluator)")
        print(f"  - 标准库: jiwer (CER) + sacrebleu (BLEU, chrF++)")
        print(f"  - 评估模式: {mode_name}")
        print(f"  - 异常输出已标准化处理")

        print(f"\n📋 各项目详细指标:")
        if item_metrics:
            sorted_items = sorted(item_metrics, key=lambda x: x.get('character_accuracy', 0), reverse=True)
            
            for item in sorted_items[:10]:  # 显示前10个项目
                item_id = item.get('item_id', 'Unknown')
                print(f"  {item_id}: CER={item['cer']:.3f}, ACC={item['character_accuracy']:.3f}, "
                      f"BLEU={item['bleu_score']:.3f}, chrF++={item['chrf_plus_plus']:.3f}")

    def save_results(self, overall_metrics: Dict, item_metrics: List[Dict], output_path: str):
        """Save evaluation results."""
        output_data = {
            'evaluation_mode': self.evaluation_mode,
            'overall_metrics': overall_metrics,
            'item_metrics': item_metrics,
            'framework_version': 'unified_evaluation_framework_v1.0'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 评估结果已保存到: {output_path}")

    def run_evaluation(self) -> Tuple[Dict, List[Dict]]:
        """Run the full evaluation pipeline."""
        print(f"\n🚀 开始运行统一评估框架...")
        
        # Load data
        self.load_reference_data()
        self.load_prediction_data()
        
        # Execute logic based on evaluation mode
        if self.evaluation_mode == "segmented":
            matches = self.match_predictions_to_references()
            item_metrics = self.calculate_segmented_evaluation(matches)
        else:  # text_only
            item_metrics = self.calculate_text_only_evaluation()
        
        # Compute overall metrics
        overall_metrics = self.calculate_overall_metrics(item_metrics)
        
        # Display results
        self.display_results(overall_metrics, item_metrics)
        
        return overall_metrics, item_metrics

    @abstractmethod
    def calculate_segmented_evaluation(self, matches: List[Dict]) -> List[Dict]:
        """Calculate segmented evaluation results."""
        pass

    @abstractmethod
    def calculate_text_only_evaluation(self) -> List[Dict]:
        """Calculate full-text evaluation results."""
        pass
