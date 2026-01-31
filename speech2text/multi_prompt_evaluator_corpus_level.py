#!/usr/bin/env python3
"""
QwenVL multi-prompt evaluation script - corpus-level standalone v2
Reimplemented to match the design pattern of qwenvl_whisper_fusion_2.py.

Key features:
1. No Ground Truth dependency: uniformly extract one frame per second, not by subtitle timestamps
2. Deduplication: keep only one from consecutive identical results
3. Corpus-level evaluation: compute metrics on the fully concatenated text, not per-item averages
4. Caching and processing aligned with qwenvl_whisper_fusion_2.py

Capabilities:
1. Run end-to-end generation and evaluation for each prompt
2. Compare prompt performance (corpus-level metrics)
3. Produce detailed comparison reports
4. Identify the best prompt

Usage:
    # Evaluate all prompts
    python multi_prompt_evaluator_corpus_level_v2.py

    # Evaluate specific prompts
    python multi_prompt_evaluator_corpus_level_v2.py --prompts ocr_focused subtitle_specific

    # Limit number of videos
    python multi_prompt_evaluator_corpus_level_v2.py --max-videos 3
"""

import argparse
import json
import time
import pandas as pd
import sacrebleu
import jiwer
import cv2
import base64
import requests
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import re


def deduplicate_consecutive_predictions(predictions: List[str]) -> List[str]:
    """Deduplicate consecutive identical predictions."""
    if not predictions:
        return []
    
    deduplicated = [predictions[0]]
    for i in range(1, len(predictions)):
        # Simple text similarity check
        if predictions[i].strip() != predictions[i-1].strip():
            deduplicated.append(predictions[i])
    
    return deduplicated


class CorpusLevelEvaluator:
    """Corpus-level evaluator aligned with qwenvl_whisper_fusion_2.py."""
    
    def __init__(self, video_dir: str, subtitle_dir: str = None):
        self.video_dir = Path(video_dir)
        self.subtitle_dir = Path(subtitle_dir) if subtitle_dir else self.video_dir
        
        # Cache directories aligned with qwenvl_whisper_fusion_2.py
        self.frames_cache_dir = Path("speech2text/frames_cache_hybrid")
        self.frames_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # QwenVL results cache directory
        self.qwenvl_cache_dir = Path("speech2text/qwenvl_cache_hybrid_multi_prompt_3090")
        self.qwenvl_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # API configuration
        self.qwenvl_api_url = "http://localhost:8000/v1/chat/completions"
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.frame_interval = 1.0  # Extract one frame per second
        
        self.logger.info(f"初始化Corpus-level评估器")
        self.logger.info(f"视频目录: {self.video_dir}")
        self.logger.info(f"字幕目录: {self.subtitle_dir}")
        self.logger.info(f"帧缓存目录: {self.frames_cache_dir}")
        self.logger.info(f"QwenVL缓存目录: {self.qwenvl_cache_dir}")
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("CorpusLevelEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def get_video_info(self, video_path: str) -> Tuple[float, float, int]:
        """Get basic video info."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        return duration, fps, frame_count

    def extract_frames_uniform(self, video_path: str, video_name: str) -> List[Tuple[int, float, str]]:
        """
        Extract frames at a fixed interval (with caching), aligned with
        qwenvl_whisper_fusion_2.py.
        
        Returns:
            List of (frame_index, timestamp, frame_image_path)
        """
        # Check for cache
        video_cache_dir = self.frames_cache_dir / video_name
        if self._check_frame_cache_exists(video_name):
            self.logger.info(f"从缓存加载视频帧: {video_name}")
            return self._load_frames_from_cache(video_name)
        
        self.logger.info(f"提取视频帧: {video_name}")
        video_cache_dir.mkdir(exist_ok=True)
        
        # Get video info
        duration, fps, frame_count = self.get_video_info(video_path)
        
        cap = cv2.VideoCapture(video_path)
        frames_info = []
        
        # Extract frames at fixed intervals
        current_time = 0.0
        while current_time < duration:
            frame_index = int(current_time * fps)
            if frame_index >= frame_count:
                break
                
            # Seek to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Save frame to cache
                frame_filename = f"frame_{frame_index}_{current_time:.2f}.jpg"
                frame_path = video_cache_dir / frame_filename
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                frames_info.append((frame_index, current_time, str(frame_path)))
                self.logger.debug(f"提取帧: {frame_index} @ {current_time:.2f}s")
            
            current_time += self.frame_interval
        
        cap.release()
        self.logger.info(f"共提取 {len(frames_info)} 帧到缓存")
        return frames_info
    
    def _check_frame_cache_exists(self, video_name: str) -> bool:
        """Check whether frame cache exists."""
        video_cache_dir = self.frames_cache_dir / video_name
        if not video_cache_dir.exists():
            return False
        
        frame_files = list(video_cache_dir.glob("frame_*.jpg"))
        return len(frame_files) > 0
    
    def _load_frames_from_cache(self, video_name: str) -> List[Tuple[int, float, str]]:
        """Load frame info from cache."""
        video_cache_dir = self.frames_cache_dir / video_name
        frame_files = sorted(video_cache_dir.glob("frame_*.jpg"))
        
        frames_info = []
        for frame_file in frame_files:
            # Parse frame info from filename: frame_{index}_{timestamp}.jpg
            name_parts = frame_file.stem.split('_')
            if len(name_parts) >= 3:
                try:
                    frame_index = int(name_parts[1])
                    timestamp = float(name_parts[2])
                    frames_info.append((frame_index, timestamp, str(frame_file)))
                except (ValueError, IndexError):
                    continue
        
        return frames_info
    
    def _check_qwenvl_cache_exists(self, video_name: str, prompt_name: str) -> bool:
        """Check whether the QwenVL cache exists."""
        cache_file = self.qwenvl_cache_dir / f"{video_name}_{prompt_name}_qwenvl_results.json"
        return cache_file.exists()
    
    def _save_qwenvl_cache(self, video_name: str, prompt_name: str, subtitles: List[Tuple[float, str]]):
        """Save QwenVL results to cache."""
        cache_file = self.qwenvl_cache_dir / f"{video_name}_{prompt_name}_qwenvl_results.json"
        cache_data = {
            "video_name": video_name,
            "prompt_name": prompt_name,
            "timestamp": datetime.now().isoformat(),
            "subtitles": subtitles,
            "frame_interval": self.frame_interval
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"QwenVL结果已缓存: {cache_file}")
        except Exception as e:
            self.logger.error(f"保存QwenVL缓存失败: {e}")
    
    def _load_qwenvl_cache(self, video_name: str, prompt_name: str) -> List[Tuple[float, str]]:
        """Load QwenVL results from cache."""
        cache_file = self.qwenvl_cache_dir / f"{video_name}_{prompt_name}_qwenvl_results.json"
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check whether the cache matches current configuration
            if cache_data.get("frame_interval") == self.frame_interval:
                subtitles = [(float(ts), text) for ts, text in cache_data["subtitles"]]
                self.logger.info(f"从缓存加载QwenVL结果: {len(subtitles)} 个字幕片段")
                return subtitles
            else:
                self.logger.warning(f"QwenVL缓存配置不匹配，将重新生成")
                return []
                
        except Exception as e:
            self.logger.error(f"加载QwenVL缓存失败: {e}")
            return []

    def clear_qwenvl_cache(self, video_name: str = None, prompt_name: str = None):
        """
        Clear QwenVL cache.

        Args:
            video_name: Specific video name.
            prompt_name: Specific prompt name.
            If both are None, clear all caches.
        """
        if video_name and prompt_name:
            cache_file = self.qwenvl_cache_dir / f"{video_name}_{prompt_name}_qwenvl_results.json"
            if cache_file.exists():
                cache_file.unlink()
                self.logger.info(f"已清理QwenVL缓存: {video_name}_{prompt_name}")
        else:
            # Clear all caches
            for cache_file in self.qwenvl_cache_dir.glob("*_qwenvl_results.json"):
                cache_file.unlink()
            self.logger.info("已清理所有QwenVL缓存")

    def extract_subtitles_with_qwenvl(self, frame_paths: List[Tuple[int, float, str]], video_name: str, prompt_template: str, prompt_name: str) -> List[Tuple[float, str]]:
        """
        Extract subtitles with QwenVL (with caching), aligned with
        qwenvl_whisper_fusion_2.py.
        
        Args:
            frame_paths: List of (frame_index, timestamp, frame_image_path)
            video_name: Video name.
            prompt_template: Prompt template.
            prompt_name: Prompt name (for caching).
            
        Returns:
            List of (timestamp, subtitle_text)
        """
        # Check cache first
        if self._check_qwenvl_cache_exists(video_name, prompt_name):
            self.logger.info(f"发现QwenVL缓存，直接加载: {video_name}_{prompt_name}")
            cached_subtitles = self._load_qwenvl_cache(video_name, prompt_name)
            if cached_subtitles:  # Return directly if cache is valid
                return cached_subtitles
        
        # Cache missing or invalid; call QwenVL
        self.logger.info(f"使用QwenVL提取 {len(frame_paths)} 帧字幕，prompt: {prompt_name}")
        
        from tqdm import tqdm
        subtitles = []
        for frame_index, timestamp, frame_path in tqdm(frame_paths, desc=f"QwenVL字幕提取", unit="帧"):
            try:
                # Read frame image and encode
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                # Convert to RGB and encode as JPEG
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                success, buffer = cv2.imencode('.jpg', frame_rgb)
                if not success:
                    continue
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                # Call QwenVL API
                subtitle_text = self._call_qwenvl_api(image_base64, prompt_template)
                # Save all results (including "no subtitle"); dedup happens later
                subtitles.append((timestamp, subtitle_text.strip()))
                self.logger.debug(f"帧 {frame_index} @ {timestamp:.2f}s: {subtitle_text}")
            except Exception as e:
                self.logger.error(f"处理帧 {frame_index} 时出错: {e}")
                continue
        # Save results to cache
        self._save_qwenvl_cache(video_name, prompt_name, subtitles)
        self.logger.info(f"QwenVL成功提取 {len(subtitles)} 个字幕片段")
        return subtitles

    def _call_qwenvl_api(self, image_base64: str, prompt_template: str) -> str:
        """Call the QwenVL API, aligned with qwenvl_whisper_fusion_2.py."""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_template
                        }
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(
                self.qwenvl_api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            self.logger.error(f"QwenVL API调用失败: {e}")
            return ""

    def deduplicate_adjacent_subtitles(self, subtitles: List[Tuple[float, str]]) -> List[str]:
        """
        Deduplicate adjacent subtitles and extract text, aligned with
        qwenvl_whisper_fusion_2.py.
        
        Args:
            subtitles: [(timestamp, subtitle_text), ...]
            
        Returns:
            List of deduplicated texts.
        """
        if not subtitles:
            return []
        
        # Key fix: sort by timestamp to avoid cache ordering issues
        subtitles.sort(key=lambda x: x[0])
        self.logger.info(f"按时间戳排序了 {len(subtitles)} 个字幕片段")
        
        merged_texts = []
        prev_text = ""
        
        for timestamp, text in subtitles:
            # Filter out "no subtitle" and empty text
            # if text and text.strip() not in ['无字幕', '[空]', '']:
            #     # Simple dedup: keep if different from previous text
            #     if text != prev_text:
            #         merged_texts.append(text)
            #         prev_text = text
            #         self.logger.debug(f"Add new subtitle @ {timestamp:.2f}s: {text}")
            #     else:
            #         self.logger.debug(f"Skip duplicate subtitle @ {timestamp:.2f}s: {text}")

            if text != prev_text:
                merged_texts.append(text)
                prev_text = text
                self.logger.debug(f"Add new subtitle @ {timestamp:.2f}s: {text}")
            else:
                self.logger.debug(f"Skip duplicate subtitle @ {timestamp:.2f}s: {text}")  
                      
        self.logger.info(f"字幕去重: {len(subtitles)} -> {len(merged_texts)} 个独特片段")
        
        # Terminal preview: show the first 10 subtitles after sorting
        if merged_texts:
            preview_count = min(10, len(merged_texts))
            preview_text = " | ".join(merged_texts[:preview_count])
            self.logger.info(f"去重后字幕预览（前{preview_count}条）: {preview_text}")
        
        return merged_texts

    def get_complete_subtitle_text(self, video_name: str) -> str:
        """Get full text directly from subtitle files (for corpus-level eval)."""
        base_name = video_name.replace('.mp4', '')
        
        # Try SRT subtitle file formats
        possible_files = [
            self.subtitle_dir / f"{base_name}新.srt",
            self.subtitle_dir / f"{base_name}.srt"
        ]
        
        for subtitle_path in possible_files:
            if subtitle_path.exists():
                return self._parse_srt_to_text(subtitle_path)
        
        self.logger.warning(f"未找到视频 {video_name} 的字幕文件")
        return ""
    
    def _parse_srt_to_text(self, srt_path: Path) -> str:
        """Parse an SRT file into plain text."""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip index lines
                if line.isdigit():
                    continue
                # Skip timestamp lines
                if '-->' in line:
                    continue
                # Skip empty lines
                if not line:
                    continue
                
                # Remove speaker identifiers
                line = re.sub(r'^\d+\s+', '', line)
                text_lines.append(line)
            
            return " ".join(text_lines)
            
        except Exception as e:
            self.logger.error(f"解析SRT文件失败 {srt_path}: {e}")
            return ""

    def calculate_corpus_level_metrics_direct(self, prediction_text: str, reference_text: str, system_name: str) -> Dict:
        """Compute corpus-level metrics on concatenated text."""
        self.logger.info(f"计算 {system_name} 的 corpus-level 指标...")
        
        if not prediction_text or not reference_text:
            return {
                'bleu_score': 0.0,
                'chrf_score': 0.0,
                'cer': 1.0,
                'character_accuracy': 0.0,
                'composite_score': 0.0,
                'total_pairs': 0
            }
        
        # Log corpus-level documents
        self.logger.info(f"Corpus-level 文档:")
        self.logger.info(f"   预测文档总长度: {len(prediction_text)} 字符")
        self.logger.info(f"   参考文档总长度: {len(reference_text)} 字符")
        
        # Log the first 200 characters used for evaluation
        self.logger.info(f"   预测文本前200字符: {prediction_text[:200]}...")
        self.logger.info(f"   参考文本前200字符: {reference_text[:200]}...")
        
        # 1. BLEU (corpus-level) - treat the text as a single sentence
        bleu = sacrebleu.corpus_bleu([prediction_text], [[reference_text]], tokenize='zh')
        
        # 2. chrF++ (corpus-level)
        chrf = sacrebleu.corpus_chrf([prediction_text], [[reference_text]], word_order=2)
        
        # 3. CER (corpus-level)
        corpus_cer = jiwer.cer(reference_text, prediction_text)
        
        # 4. Character accuracy (corpus-level)
        if reference_text:
            output = jiwer.process_characters(reference_text, prediction_text)
            correct_chars = output.hits
            total_chars_in_ref = output.hits + output.substitutions + output.deletions
            corpus_char_accuracy = correct_chars / total_chars_in_ref if total_chars_in_ref > 0 else 0.0
        else:
            corpus_char_accuracy = 1.0 if not prediction_text else 0.0
        
        # 5. Composite score
        bleu_normalized = bleu.score / 100.0
        chrf_normalized = chrf.score / 100.0
        cer_accuracy = max(0.0, 1.0 - corpus_cer)
        composite_score = (bleu_normalized + chrf_normalized + cer_accuracy + corpus_char_accuracy) / 4.0
        
        metrics = {
            'bleu_score': bleu.score,
            'chrf_score': chrf.score,
            'cer': corpus_cer,
            'character_accuracy': corpus_char_accuracy,
            'composite_score': composite_score,
            'bleu_normalized': bleu_normalized,
            'chrf_normalized': chrf_normalized,
            'cer_accuracy': cer_accuracy,
            'total_pairs': 1,  # Whole corpus as one comparison pair
            'total_pred_chars': len(prediction_text),
            'total_ref_chars': len(reference_text)
        }
        
        self.logger.info(f"{system_name} corpus-level 指标计算完成")
        self.logger.info(f"   - BLEU: {metrics['bleu_score']:.4f}")
        self.logger.info(f"   - chrF++: {metrics['chrf_score']:.4f}")
        self.logger.info(f"   - CER: {metrics['cer']:.4f}")
        self.logger.info(f"   - 字符准确率: {metrics['character_accuracy']:.4f}")
        self.logger.info(f"   - 综合指标: {composite_score:.4f}")
        
        return metrics

    def evaluate_single_prompt(self, prompt_name: str, prompt_template: str, max_videos: int = 0) -> Dict[str, Any]:
        """Evaluate a single prompt."""
        self.logger.info(f"开始评估 Prompt: {prompt_name}")
        self.logger.info("=" * 60)
        
        # Get video list
        video_files = list(self.video_dir.glob('*.mp4'))
        if max_videos > 0:
            video_files = video_files[:max_videos]
        
        self.logger.info(f"将处理 {len(video_files)} 个视频")
        
        all_predictions = []
        video_details = []
        
        start_time = time.time()
        
        for video_file in video_files:
            self.logger.info(f"处理视频: {video_file.name}")
            
            video_base_name = video_file.stem
            
            # Extract frames
            frame_paths = self.extract_frames_uniform(str(video_file), video_base_name)
            self.logger.info(f"提取了 {len(frame_paths)} 帧")
            
            # Extract subtitles with QwenVL (cached)
            subtitles = self.extract_subtitles_with_qwenvl(
                frame_paths, video_base_name, prompt_template, prompt_name
            )
            
            # Apply deduplication and extract text
            video_predictions = self.deduplicate_adjacent_subtitles(subtitles)
            self.logger.info(f"去重处理: {len(subtitles)} -> {len(video_predictions)} 条")
            
            all_predictions.extend(video_predictions)
            
            # Collect per-video details
            video_details.append({
                'video_name': video_file.name,
                'original_prediction_count': len(subtitles),
                'deduplicated_prediction_count': len(video_predictions),
                'predictions': " | ".join(video_predictions)
            })
        
        processing_time = time.time() - start_time
        self.logger.info(f"{prompt_name} 处理完成，耗时: {processing_time:.2f} 秒")
        
        # Concatenate predictions across all videos
        corpus_prediction_text = " ".join(all_predictions)
        
        # Fetch and concatenate references for all videos
        self.logger.info("正在准备 Corpus-level 参考文本...")
        all_references = []
        for video_file in video_files:
            ref_text = self.get_complete_subtitle_text(video_file.name)
            if ref_text:
                all_references.append(ref_text)
        corpus_reference_text = " ".join(all_references)
        self.logger.info("参考文本准备完成")
        
        # Compute corpus-level metrics
        metrics = self.calculate_corpus_level_metrics_direct(
            corpus_prediction_text, 
            corpus_reference_text,
            prompt_name
        )
        
        return {
            "prompt_name": prompt_name,
            "metrics": metrics,
            "total_videos": len(video_files),
            "processing_time": processing_time,
            "video_details": video_details
        }


def main():
    """Main entry point."""
    # Prompt configuration
    PROMPT_CANDIDATES = {
        "simple_direct": "识别图片中的字幕文字：",
        "few_shot_simple": "从图片中提取字幕。例如，如果图片字幕是“你好世界”，你应该只输出“你好世界”。现在，请处理这张图片。",
        "ocr_focused": "你是一个专业的OCR文字识别引擎。你的任务是精确地转录这张图片中的所有字幕文本，确保100%的准确性。请识别图片中的字幕文字，只返回识别到的文字内容，不要添加任何解释。如果没有字幕，返回'无字幕'。",
        "subtitle_specific": "请识别这张图片中的字幕文字。要求：1)只返回字幕内容，不要其他描述 2)保持原文格式 3)如果没有字幕返回'无字幕'",
        "context_aware_new": "这是一个视频截图，请识别其中的中文字幕内容。只返回字幕文字，忽略其他图像元素。如果没有字幕，请输出[空]。",
        "chain_of_thought": "请分步执行任务：1. 定位图片中字幕的位置。2. 忽略背景中的干扰元素。3. 仔细识别并转录你找到的字幕。最后，只输出最终转录的字幕文本。",
        "robustness_instruction": "这张图片的背景可能很复杂，字幕可能有多种颜色或被轻微遮挡。请尽最大努力，精确地提取出所有可见的字幕文本。"
    }
    
    parser = argparse.ArgumentParser(
        description='QwenVL 多 Prompt 循环评估脚本 - Corpus-level 独立版本 v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
重要特性:
  - 均匀提取: 每秒提取一帧，不依赖 ground truth
  - 去重处理: 相邻相同预测结果自动去重  
  - Corpus-level 评估: 整体拼接计算指标，而非逐项平均
  - 完全向 qwenvl_whisper_fusion_2.py 看齐的缓存和处理机制

可用的 Prompt:
{chr(10).join(f"  - {name}: {template[:60]}..." for name, template in PROMPT_CANDIDATES.items())}

使用示例:
  # 评估所有 prompt
  python multi_prompt_evaluator_corpus_level_v2.py
  
  # 评估指定的 prompt
  python multi_prompt_evaluator_corpus_level_v2.py --prompts simple_direct ocr_focused
  
  # 限制测试视频数量（快速测试）
  python multi_prompt_evaluator_corpus_level_v2.py --max-videos 2
        """
    )
    
    parser.add_argument('--prompts', nargs='+', 
                       choices=list(PROMPT_CANDIDATES.keys()),
                       help='指定要评估的prompt名称')
    
    parser.add_argument('--max-videos', type=int, default=0,
                       help='限制最大视频数量 (0=不限制)')
    
    parser.add_argument('--video-dir', default="data/闪婚幸运草的命中注定",
                       help='视频目录路径')
    
    parser.add_argument('--subtitle-dir', default="data/闪婚幸运草的命中注定/带角色标注的字幕", 
                       help='字幕目录路径')
    
    parser.add_argument('--clear-cache', action='store_true',
                       help='清理所有QwenVL缓存，强制重新处理')
    
    args = parser.parse_args()
    
    print("🚀 QwenVL 多 Prompt 循环评估脚本 - Corpus-level 独立版本 v2")
    print("🔬 每秒均匀提取 + 去重 + 整体拼接评估")
    print("⚠️  注意: 不依赖 ground truth，只在最终评估时使用参考文本")
    
    # Determine prompts to evaluate
    if args.prompts:
        prompts_to_evaluate = {name: PROMPT_CANDIDATES[name] for name in args.prompts if name in PROMPT_CANDIDATES}
    else:
        prompts_to_evaluate = PROMPT_CANDIDATES
    
    print(f"📋 将评估 {len(prompts_to_evaluate)} 个 Prompt")
    
    # Create evaluator
    evaluator = CorpusLevelEvaluator(args.video_dir, args.subtitle_dir)
    
    # Clear cache if requested
    if args.clear_cache:
        evaluator.clear_qwenvl_cache()
        print("🗑️ 已清理所有QwenVL缓存")
    
    # Create output directory
    output_dir = Path("speech2text/corpus_level_evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run evaluation
    all_results = []
    total_start_time = time.time()
    
    for i, (prompt_name, prompt_template) in enumerate(prompts_to_evaluate.items(), 1):
        print(f"\n{'='*20} [{i}/{len(prompts_to_evaluate)}] {'='*20}")
        
        try:
            result = evaluator.evaluate_single_prompt(prompt_name, prompt_template, args.max_videos)
            all_results.append(result)
            
            print(f"✅ {prompt_name} 评估完成 (预测结果已缓存到 {evaluator.qwenvl_cache_dir})")
            
        except Exception as e:
            print(f"❌ {prompt_name} 评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # Generate final ranking
    if all_results:
        print(f"\n🏆 最终 Corpus-level 排序 (按综合指标):")
        print("-" * 80)
        
        sorted_results = sorted(all_results, 
                              key=lambda x: x['metrics']['composite_score'], 
                              reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            metrics = result['metrics']
            print(f"#{rank:2d}. {result['prompt_name']:<25} "
                  f"综合: {metrics['composite_score']:.4f} "
                  f"BLEU: {metrics['bleu_score']:.4f} "
                  f"chrF++: {metrics['chrf_score']:.4f} "
                  f"CER: {metrics['cer']:.4f}")
        
        # Save summary
        summary_data = []
        for rank, result in enumerate(sorted_results, 1):
            metrics = result['metrics']
            summary_data.append({
                'rank': rank,
                'prompt_name': result['prompt_name'],
                'composite_score': metrics['composite_score'],
                'bleu_score': metrics['bleu_score'],
                'chrf_score': metrics['chrf_score'],
                'cer': metrics['cer'],
                'character_accuracy': metrics['character_accuracy'],
                'total_videos': result['total_videos'],
                'processing_time': result['processing_time'],
                'methodology': 'corpus_level_uniform_extraction_with_deduplication'
            })
        
        summary_csv = output_dir / f"corpus_level_ranking_{timestamp}.csv"
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_csv, index=False, encoding='utf-8')
        
        print(f"\n📊 结果已保存到: {output_dir}")
        print(f"  - corpus_level_ranking_{timestamp}.csv: 最终排序")
        print(f"  - QwenVL预测结果缓存目录: {evaluator.qwenvl_cache_dir}")
        print(f"  - 视频帧缓存目录: {evaluator.frames_cache_dir}")
    
    print(f"\n🎉 评估完成! 总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    main()
