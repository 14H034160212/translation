#!/usr/bin/env python3
"""
Whisper Multi-Model Evaluator
Evaluates and compares multiple Whisper model sizes using unified evaluation framework.
"""

import os
import sys
import logging
import time
import whisper
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import csv
import re

sys.path.append(str(Path(__file__).parent))

from unified_evaluation_framework import EvaluationMetrics

@dataclass
class WhisperResult:
    """Whisper transcription result."""
    video_file: str
    full_transcription: str
    confidence: float
    processing_time: float
    error: Optional[str] = None

class WhisperModelEvaluator:
    """Whisper model evaluator for MP4 files."""
    
    def __init__(self):
        self.evaluation_metrics = EvaluationMetrics()
        self.setup_logging()
        
        self.MODELS_TO_TEST = {
            "tiny": {"name": "tiny", "description": "Whisper Tiny"},
            "base": {"name": "base", "description": "Whisper Base"},
            "small": {"name": "small", "description": "Whisper Small"},
            "medium": {"name": "medium", "description": "Whisper Medium"},
            "large": {"name": "large", "description": "Whisper Large"}
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_test_data(self, video_dir: str, max_videos: int = 10) -> List[str]:
        if not Path(video_dir).is_absolute():
            video_path = Path(__file__).parent.parent / video_dir
        else:
            video_path = Path(video_dir)
        
        mp4_files = sorted(list(video_path.glob("*.mp4")))
        
        if max_videos > 0:
            mp4_files = mp4_files[:max_videos]
            
        video_files = [str(f) for f in mp4_files]
        self.logger.info(f"Loaded {len(video_files)} MP4 files for testing")
        return video_files
    
    def transcribe_with_whisper(self, model: Any, video_file: str) -> WhisperResult:
        try:
            start_time = time.time()
            
            result = model.transcribe(
                video_file, 
                language='zh', 
                # temperature=0.0
                )
            
            processing_time = time.time() - start_time
            
            # Extract text from segments and join with spaces
            if 'segments' in result and result['segments']:
                segment_texts = []
                for seg in result['segments']:
                    text = seg.get('text', '').strip()
                    if text:
                        segment_texts.append(text)
                full_text = ' '.join(segment_texts)
            else:
                # If there are no segments, use the full text
                full_text = result.get('text', '').strip()
            
            # Compute average confidence
            if 'segments' in result and result['segments']:
                confidences = []
                for seg in result['segments']:
                    if 'avg_logprob' in seg:
                        confidence = np.exp(seg['avg_logprob'])
                        confidences.append(confidence)
                avg_confidence = np.mean(confidences) if confidences else 0.5
            else:
                avg_confidence = 0.5
            
            return WhisperResult(
                video_file=video_file,
                full_transcription=full_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Transcription failed {video_file}: {e}")
            return WhisperResult(
                video_file=video_file,
                full_transcription="",
                confidence=0.0,
                processing_time=0.0,
                error=str(e)
            )
    
    def load_ground_truth(self, video_file: str, ground_truth_dir: str) -> str:
        video_name = Path(video_file).stem
        
        if not Path(ground_truth_dir).is_absolute():
            gt_dir = Path(__file__).parent.parent / ground_truth_dir
        else:
            gt_dir = Path(ground_truth_dir)
            
        gt_file = gt_dir / f"{video_name}新.srt"
        
        if not gt_file.exists():
            gt_file = gt_dir / f"{video_name}.srt"
        
        if gt_file.exists():
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_content = f.read()
                
                gt_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', gt_content)
                gt_text = re.sub(r'\n\d+\s+', '\n', gt_text)
                gt_text = re.sub(r'^\d+\s+', '', gt_text)
                gt_text = re.sub(r'\n+', ' ', gt_text).strip()
                
                return gt_text
            except Exception as e:
                self.logger.error(f"Failed to read ground truth: {e}")
                return ""
        else:
            self.logger.warning(f"Ground truth file not found: {gt_file}")
            return ""
    
    def run_single_model_evaluation(self, model_name: str, video_dir: str = "data/闪婚幸运草的命中注定", 
                                   ground_truth_dir: str = "data/闪婚幸运草的命中注定/带角色标注的字幕",
                                   max_videos: int = 10) -> Dict[str, Any]:
        self.logger.info(f"Starting evaluation for model: {model_name}")
        
        output_dir = Path(f"speech2text/whisper_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model = whisper.load_model(self.MODELS_TO_TEST[model_name]["name"])
            self.logger.info(f"Model loaded successfully: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return {"error": f"Failed to load model: {e}"}
        
        video_files = self.load_test_data(video_dir, max_videos)
        
        results_file = output_dir / f"whisper_{model_name}_results.csv"
        
        results = []
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'video_file', 'model_name', 'full_transcription', 'ground_truth',
                'cer', 'character_accuracy', 'bleu_score', 'chrf_score',
                'confidence', 'processing_time', 'status'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, video_file in enumerate(video_files, 1):
                self.logger.info(f"Processing video {i}/{len(video_files)}: {Path(video_file).name}")
                
                gt_text = self.load_ground_truth(video_file, ground_truth_dir)
                if not gt_text:
                    self.logger.warning(f"Skipping {Path(video_file).name}: No ground truth available")
                    # Record the skip reason without affecting metric calculations
                    row = {
                        'video_file': Path(video_file).name,
                        'model_name': f"whisper_{model_name}",
                        'full_transcription': '',
                        'ground_truth': '',
                        'cer': '',
                        'character_accuracy': '',
                        'bleu_score': '',
                        'chrf_score': '',
                        'confidence': '',
                        'processing_time': '',
                        'status': 'SKIPPED: 无标准字幕'
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    results.append(row)
                    continue
                
                whisper_result = self.transcribe_with_whisper(model, video_file)
                
                if whisper_result.error:
                    self.logger.error(f"Transcription failed: {whisper_result.error}")
                    row = {
                        'video_file': Path(video_file).name,
                        'model_name': f"whisper_{model_name}",
                        'full_transcription': '',
                        'ground_truth': gt_text,
                        'cer': '',
                        'character_accuracy': '',
                        'bleu_score': '',
                        'chrf_score': '',
                        'confidence': 0.0,
                        'processing_time': 0.0,
                        'status': f'ERROR: {whisper_result.error}'
                    }
                else:
                    if whisper_result.full_transcription:
                        metrics = self.evaluation_metrics.calculate_all_metrics(
                            whisper_result.full_transcription, 
                            gt_text
                        )
                        
                        row = {
                            'video_file': Path(video_file).name,
                            'model_name': f"whisper_{model_name}",
                            'full_transcription': whisper_result.full_transcription,
                            'ground_truth': gt_text,
                            'cer': metrics['cer'],
                            'character_accuracy': metrics['character_accuracy'],
                            'bleu_score': metrics['bleu_score'],
                            'chrf_score': metrics['chrf_plus_plus'],
                            'confidence': whisper_result.confidence,
                            'processing_time': whisper_result.processing_time,
                            'status': 'SUCCESS'
                        }
                    else:
                        # Handle empty transcription
                        row = {
                            'video_file': Path(video_file).name,
                            'model_name': f"whisper_{model_name}",
                            'full_transcription': '',
                            'ground_truth': gt_text,
                            'cer': '',
                            'character_accuracy': '',
                            'bleu_score': '',
                            'chrf_score': '',
                            'confidence': whisper_result.confidence,
                            'processing_time': whisper_result.processing_time,
                            'status': 'ERROR: 转录结果为空'
                        }
                
                writer.writerow(row)
                csvfile.flush()  # Flush immediately
                results.append(row)
        
        self.logger.info(f"✅ 模型 {model_name} 评估完成，结果保存到: {results_file}")
        
        # Compute summary stats using successful evaluations only
        successful_results = [r for r in results if r['status'] == 'SUCCESS' and isinstance(r['cer'], (int, float))]
        skipped_results = [r for r in results if 'SKIPPED' in r['status']]
        error_results = [r for r in results if 'ERROR' in r['status']]
        
        self.logger.info(f"评估统计: 总视频 {len(results)}, 成功评估 {len(successful_results)}, 跳过 {len(skipped_results)}, 失败 {len(error_results)}")
        
        if successful_results:
            avg_metrics = {
                'cer': np.mean([r['cer'] for r in successful_results]),
                'character_accuracy': np.mean([r['character_accuracy'] for r in successful_results]),
                'bleu_score': np.mean([r['bleu_score'] for r in successful_results]),
                'chrf_score': np.mean([r['chrf_score'] for r in successful_results]),
                'confidence': np.mean([r['confidence'] for r in successful_results]),
                'processing_time': np.mean([r['processing_time'] for r in successful_results])
            }
        else:
            self.logger.warning("没有成功的评估结果可用于统计")
            avg_metrics = {
                'cer': float('nan'),
                'character_accuracy': float('nan'),
                'bleu_score': float('nan'),
                'chrf_score': float('nan'),
                'confidence': float('nan'),
                'processing_time': float('nan')
            }
        
        return {
            'model_name': model_name,
            'results_file': str(results_file),
            'total_videos': len(results),
            'successful_evaluations': len(successful_results),
            'skipped_videos': len(skipped_results),
            'failed_videos': len(error_results),
            'avg_metrics': avg_metrics
        }
    
    def run_all_model_evaluations(self, models_to_test: List[str] = None, **kwargs):
        """Run evaluations for all models."""
        if models_to_test is None:
            models_to_test = ["base", "small"]  # Default to these two models
        
        self.logger.info(f"开始评估 {len(models_to_test)} 个Whisper模型")
        
        all_results = []
        for model_name in models_to_test:
            if model_name not in self.MODELS_TO_TEST:
                self.logger.error(f"未知模型: {model_name}")
                continue
                
            result = self.run_single_model_evaluation(model_name, **kwargs)
            all_results.append(result)
            
            # Print current model results
            if 'error' not in result:
                metrics = result['avg_metrics']
                self.logger.info(f"""
模型 {model_name} 评估结果:
- 总视频数: {result['total_videos']}
- 成功评估: {result['successful_evaluations']}
- 跳过视频: {result['skipped_videos']}
- 失败视频: {result['failed_videos']}
- CER: {metrics['cer']:.4f}
- 字符准确率: {metrics['character_accuracy']:.4f}
- BLEU: {metrics['bleu_score']:.4f}
- chrF++: {metrics['chrf_score']:.4f}
- 平均置信度: {metrics['confidence']:.4f}
- 平均处理时间: {metrics['processing_time']:.2f}秒
                """)
        
        # Print summary results
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 所有模型评估完成！")
        self.logger.info("="*60)
        
        for result in all_results:
            if 'error' not in result:
                metrics = result['avg_metrics']
                if not np.isnan(metrics['cer']):  # Only show metrics when valid
                    self.logger.info(f"""
{result['model_name']}:
  评估统计: {result['successful_evaluations']}/{result['total_videos']} 成功
  CER: {metrics['cer']:.4f} | 准确率: {metrics['character_accuracy']:.4f}
  BLEU: {metrics['bleu_score']:.4f} | chrF++: {metrics['chrf_score']:.4f}
  结果文件: {result['results_file']}
                    """)
                else:
                    self.logger.info(f"""
{result['model_name']}:
  评估统计: {result['successful_evaluations']}/{result['total_videos']} 成功
  ⚠️  没有有效的评估结果
  结果文件: {result['results_file']}
                    """)
        
        return all_results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper模型评估脚本")
    parser.add_argument("--models", nargs="+", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       default=["medium"],
                       help="要测试的模型列表")
    parser.add_argument("--video-dir", default="data/闪婚幸运草的命中注定",
                       help="视频文件目录")
    parser.add_argument("--ground-truth-dir", default="data/闪婚幸运草的命中注定/带角色标注的字幕",
                       help="字幕文件目录")
    parser.add_argument("--max-videos", type=int, default=100,
                       help="最大测试视频数量")
    
    args = parser.parse_args()
    
    evaluator = WhisperModelEvaluator()
    evaluator.run_all_model_evaluations(
        models_to_test=args.models,
        video_dir=args.video_dir,
        ground_truth_dir=args.ground_truth_dir,
        max_videos=args.max_videos
    )

if __name__ == "__main__":
    main()
