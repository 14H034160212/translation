#!/usr/bin/env python3
"""
QwenVL + Whisper Fusion System
Combines visual OCR and speech recognition for dual-modal subtitle extraction.
"""

import os
import cv2
import base64
import json
import time
import whisper
import requests
import pandas as pd
import sacrebleu
import jiwer
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import re
from rapidfuzz import fuzz

@dataclass
class FusionResult:
    """Fusion recognition result with detailed alignment information."""
    video_file: str
    total_duration: float
    
    alignment_details: List[Dict[str, Any]] 
    
    qwenvl_merged_text: str
    whisper_original: str
    final_result: str
    ground_truth: str
    
    processing_time: float
    
    stats: Dict[str, Any]
    
    qwenvl_bleu_score: float = 0.0
    qwenvl_chrf_score: float = 0.0
    qwenvl_cer: float = 0.0
    qwenvl_character_accuracy: float = 0.0
    qwenvl_composite_score: float = 0.0
    
    whisper_original_bleu_score: float = 0.0
    whisper_original_chrf_score: float = 0.0
    whisper_original_cer: float = 0.0
    whisper_original_character_accuracy: float = 0.0
    whisper_original_composite_score: float = 0.0
    
    final_bleu_score: float = 0.0
    final_chrf_score: float = 0.0
    final_cer: float = 0.0
    final_character_accuracy: float = 0.0
    final_composite_score: float = 0.0
    
    metadata: Dict[str, Any] = None

class QwenVLWhisperFusion:
    """QwenVL + Whisper fusion system with timestamp alignment strategy."""
    
    def __init__(self, 
                 qwenvl_api_url: str = "http://localhost:8000/v1/chat/completions",
                 whisper_model: str = "medium",
                 frame_interval: float = 1.0,
                 similarity_threshold: float = 60.0,
                 time_tolerance: float = 1.5):
        self.qwenvl_api_url = qwenvl_api_url
        self.whisper_model_name = whisper_model
        self.frame_interval = frame_interval
        self.similarity_threshold = similarity_threshold
        self.time_tolerance = time_tolerance
        
        # Set cache directories
        self.frames_cache_dir = Path("speech2text/frames_cache_hybrid")
        self.frames_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # QwenVL results cache directory
        self.qwenvl_cache_dir = Path("speech2text/qwenvl_cache_hybrid_no_post_3090")
        self.qwenvl_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Whisper results cache directory
        self.whisper_cache_dir = Path("speech2text/whisper_cache_medium_3090")
        self.whisper_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Whisper model
        self.logger = self._setup_logging()
        self.logger.info(f"正在加载Whisper模型: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)
        self.logger.info("Whisper模型加载完成")
        
        self.logger.info(f"融合参数: 相似度阈值={self.similarity_threshold}, 时间容忍度={self.time_tolerance}s")
        
        # QwenVL subtitle extraction prompt
        self.qwenvl_prompt = (
            "你是一个专业的OCR文字识别引擎。你的任务是精确地转录这张图片中的所有字幕文本，确保100%的准确性。请识别图片中的字幕文字，只返回识别到的文字内容，不要添加任何解释。如果没有字幕，返回'无字幕'。"
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("QwenVLWhisperFusion")
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
        Extract frames at fixed intervals (with caching).
        
        Returns:
            List of (frame_index, timestamp, frame_image_path)
        """
        # Check whether cache exists
        video_cache_dir = self.frames_cache_dir / video_name
        if self._check_cache_exists(video_name):
            self.logger.info(f"Loading cached video frames: {video_name}")
            return self._load_frames_from_cache(video_name)
        
        self.logger.info(f"Extracting video frames: {video_name}")
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
                cv2.imwrite(str(frame_path), frame)
                
                frames_info.append((frame_index, current_time, str(frame_path)))
                self.logger.debug(f"Extracted frame: {frame_index} @ {current_time:.2f}s")
            
            current_time += self.frame_interval
        
        cap.release()
        self.logger.info(f"Extracted {len(frames_info)} frames to cache")
        return frames_info
    
    def _check_cache_exists(self, video_name: str) -> bool:
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
    
    def _check_qwenvl_cache_exists(self, video_name: str) -> bool:
        """Check whether QwenVL results cache exists."""
        cache_file = self.qwenvl_cache_dir / f"{video_name}_qwenvl_results.json"
        return cache_file.exists()
    
    def _save_qwenvl_cache(self, video_name: str, subtitles: List[Tuple[float, str]]):
        """Save QwenVL results to cache."""
        cache_file = self.qwenvl_cache_dir / f"{video_name}_qwenvl_results.json"
        cache_data = {
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(),
            "subtitles": subtitles,
            "frame_interval": self.frame_interval
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"QwenVL results cached: {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save QwenVL cache: {e}")
    
    def _load_qwenvl_cache(self, video_name: str) -> List[Tuple[float, str]]:
        """Load QwenVL results from cache."""
        cache_file = self.qwenvl_cache_dir / f"{video_name}_qwenvl_results.json"
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check whether cache matches the current configuration
            if cache_data.get("frame_interval") == self.frame_interval:
                subtitles = [(float(ts), text) for ts, text in cache_data["subtitles"]]
                self.logger.info(f"Loaded QwenVL cache: {len(subtitles)} subtitle segments")
                return subtitles
            else:
                self.logger.warning("QwenVL cache config mismatch; regenerating")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to load QwenVL cache: {e}")
            return []
    
    def clear_qwenvl_cache(self, video_name: str = None):
        """
        Clear QwenVL cache.
        
        Args:
            video_name: Specific video name; if None, clear all caches.
        """
        if video_name:
            cache_file = self.qwenvl_cache_dir / f"{video_name}_qwenvl_results.json"
            if cache_file.exists():
                cache_file.unlink()
                self.logger.info(f"Cleared QwenVL cache: {video_name}")
        else:
            # Clear all caches
            for cache_file in self.qwenvl_cache_dir.glob("*_qwenvl_results.json"):
                cache_file.unlink()
            self.logger.info("Cleared all QwenVL caches")
    
    def _check_whisper_cache_exists(self, video_name: str) -> bool:
        """Check whether the Whisper results cache exists."""
        cache_file = self.whisper_cache_dir / f"{video_name}_whisper_results.json"
        return cache_file.exists()

    def _save_whisper_cache(self, video_name: str, result: Dict[str, Any]):
        """Save Whisper results to cache."""
        cache_file = self.whisper_cache_dir / f"{video_name}_whisper_results.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"Whisper results cached: {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save Whisper cache: {e}")

    def _load_whisper_cache(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Load Whisper results from cache."""
        cache_file = self.whisper_cache_dir / f"{video_name}_whisper_results.json"
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            self.logger.info(f"Loaded Whisper cache: {video_name}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to load Whisper cache: {e}")
            return None

    def clear_whisper_cache(self, video_name: str = None):
        """
        Clear Whisper cache.
        
        Args:
            video_name: Specific video name; if None, clear all caches.
        """
        if video_name:
            cache_file = self.whisper_cache_dir / f"{video_name}_whisper_results.json"
            if cache_file.exists():
                cache_file.unlink()
                self.logger.info(f"Cleared Whisper cache: {video_name}")
        else:
            for cache_file in self.whisper_cache_dir.glob("*_whisper_results.json"):
                cache_file.unlink()
            self.logger.info("Cleared all Whisper caches")
    
    def extract_subtitles_with_qwenvl(self, frame_paths: List[Tuple[int, float, str]], video_name: str) -> List[Tuple[float, str]]:
        """
        Extract subtitles with QwenVL (cached).
        
        Returns:
            List of (timestamp, subtitle_text)
        """
        # Check cache first
        if self._check_qwenvl_cache_exists(video_name):
            self.logger.info(f"Found QwenVL cache, loading: {video_name}")
            cached_subtitles = self._load_qwenvl_cache(video_name)
            if cached_subtitles:  # Return directly if cache is valid
                return cached_subtitles
        
        # Cache missing or invalid; run QwenVL
        self.logger.info(f"Extracting subtitles from {len(frame_paths)} frames with QwenVL")
        
        subtitles = []
        for i, (frame_index, timestamp, frame_path) in enumerate(frame_paths):
            try:
                # Read frame image and encode
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                success, buffer = cv2.imencode('.jpg', frame_rgb)
                if not success:
                    continue
                    
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Call QwenVL API
                subtitle_text = self._call_qwenvl_api(image_base64)
                
                # Filter empty results
                # if subtitle_text and subtitle_text.strip() not in ['[空]', '无字幕', '']:
                #     subtitles.append((timestamp, subtitle_text.strip()))
                #     self.logger.debug(f"帧 {frame_index} @ {timestamp:.2f}s: {subtitle_text}")
                subtitles.append((timestamp, subtitle_text.strip()))
                self.logger.debug(f"Frame {frame_index} @ {timestamp:.2f}s: {subtitle_text}")                
                # Progress display
                if (i + 1) % 10 == 0:
                    self.logger.info(f"QwenVL progress: {i + 1}/{len(frame_paths)}")
                    
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_index}: {e}")
                continue
        
        # Save results to cache
        self._save_qwenvl_cache(video_name, subtitles)
        
        self.logger.info(f"QwenVL extracted {len(subtitles)} subtitle segments")
        return subtitles
    
    def _call_qwenvl_api(self, image_base64: str) -> str:
        """Call the QwenVL API."""
        payload = {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
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
                            "text": self.qwenvl_prompt
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
            self.logger.error(f"QwenVL API call failed: {e}")
            return ""
    
    def deduplicate_adjacent_subtitles(self, subtitles: List[Tuple[float, str]]) -> str:
        """
        Deduplicate adjacent subtitles and merge.
        
        Args:
            subtitles: [(timestamp, subtitle_text), ...]
            
        Returns:
            Merged deduplicated text.
        """
        if not subtitles:
            return ""
        
        # Sort by timestamp
        subtitles.sort(key=lambda x: x[0])
        
        merged_texts = []
        prev_text = ""
        
        for timestamp, text in subtitles:
            # Simple dedup: keep if different from previous text
            if text != prev_text:
                merged_texts.append(text)
                prev_text = text
                self.logger.debug(f"Add new subtitle @ {timestamp:.2f}s: {text}")
            else:
                self.logger.debug(f"Skip duplicate subtitle @ {timestamp:.2f}s: {text}")
        
        # Merge all deduplicated text
        merged_text = " ".join(merged_texts)
        self.logger.info(f"Subtitle dedup: {len(subtitles)} -> {len(merged_texts)} unique segments")
        return merged_text
    
    def transcribe_with_whisper(self, audio_path: str, video_name: str) -> Dict[str, Any]:
        """
        Run Whisper ASR and return timestamped segments (cached).
        
        Args:
            audio_path: Audio file path.
            video_name: Video name (for cache key).
            
        Returns:
            Whisper result dict containing 'text' and 'segments'.
        """
        # Check cache
        if self._check_whisper_cache_exists(video_name):
            cached_result = self._load_whisper_cache(video_name)
            if cached_result:
                return cached_result

        self.logger.info(f"Starting Whisper ASR with timestamps: {video_name}")
        
        result = self.whisper_model.transcribe(
            audio_path, 
            language="zh",
            # temperature=0.2
        )
        
        # Save to cache
        self._save_whisper_cache(video_name, result)
        
        self.logger.info("Whisper ASR complete")
        return result

    def align_and_fuse_adaptive(self, 
                               qwenvl_subtitles: List[Tuple[float, str]], 
                               whisper_segments: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Adaptive fusion logic: dynamic threshold based on Whisper confidence.
        Logic: Lower Whisper confidence -> Lower OCR replacement threshold (more likely to trust OCR).
        """
        self.logger.info(f"Starting adaptive fusion: {len(qwenvl_subtitles)} subtitles")
        
        qwenvl_timed = []
        for ts, text in qwenvl_subtitles:
            qwenvl_timed.append({'start': ts, 'end': ts + self.frame_interval, 'text': text})
        
        final_segments = []
        alignment_details = []
        
        for ws in whisper_segments:
            # Calculate ASR confidence from logprob
            # avg_logprob is typically between -1.5 (low confidence) and -0.01 (high)
            logprob = ws.get('avg_logprob', -1.0)
            confidence = np.exp(logprob)
            
            # Adaptive threshold: 
            # Mapping confidence [0.0, 1.0] to threshold [45, 85]
            # More conservative: If high confidence (1.0), threshold is 85% (trust ASR)
            # If low confidence (0.0), threshold is 45% (allow more OCR substitution)
            # Static 60% corresponds to confidence 0.375
            adaptive_threshold = 45 + (confidence * 40)
            
            start_time = ws['start'] - self.time_tolerance
            end_time = ws['end'] + self.time_tolerance
            
            overlapping_qwenvl = [
                qs for qs in qwenvl_timed 
                if max(start_time, qs['start']) < min(end_time, qs['end'])
            ]
            
            detail = {
                "whisper_segment": ws,
                "confidence": float(confidence),
                "adaptive_threshold": float(adaptive_threshold),
                "decision": "whisper",
                "similarity": 0.0,
                "best_match_qwenvl": None,
                "overlapping_qwenvl": overlapping_qwenvl,
                "all_similarities": [],
                "replaced_text": None
            }
            
            if overlapping_qwenvl:
                similarities = []
                for qs in overlapping_qwenvl:
                    similarity = fuzz.ratio(ws['text'], qs['text'])
                    similarities.append({'text': qs['text'], 'sim': similarity})
                
                best_match = max(similarities, key=lambda x: x['sim'])
                detail["similarity"] = best_match['sim']
                
                if best_match['sim'] >= adaptive_threshold:
                    detail["decision"] = "qwenvl"
                    detail["replaced_text"] = best_match['text']
                    final_segments.append({**ws, 'text': best_match['text']})
                else:
                    final_segments.append(ws)
            else:
                final_segments.append(ws)
            
            alignment_details.append(detail)
            
        final_text = " ".join([s['text'].strip() for s in final_segments])
        return final_text, alignment_details

    def align_and_fuse(self, 
                       qwenvl_subtitles: List[Tuple[float, str]], 
                       whisper_segments: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Core fusion logic: timestamp alignment and text replacement.
        
        Args:
            qwenvl_subtitles: Subtitles from QwenVL [(timestamp, text), ...]
            whisper_segments: Whisper segments [{'start': float, 'end': float, 'text': str}, ...]
            
        Returns:
            (final_text, alignment_details)
        """
        self.logger.info(f"Starting fusion: {len(qwenvl_subtitles)} QwenVL subtitles, {len(whisper_segments)} Whisper segments")
        
        # Preprocess QwenVL subtitles and assign end times
        qwenvl_timed = []
        for ts, text in qwenvl_subtitles:
            qwenvl_timed.append({'start': ts, 'end': ts + self.frame_interval, 'text': text})
        
        final_segments = []
        alignment_details = []
        
        whisper_idx = 0
        while whisper_idx < len(whisper_segments):
            ws = whisper_segments[whisper_idx]
            
            # Find QwenVL subtitles overlapping in time
            # Time window: [ws['start'] - tolerance, ws['end'] + tolerance]
            start_time = ws['start'] - self.time_tolerance
            end_time = ws['end'] + self.time_tolerance
            
            overlapping_qwenvl = [
                qs for qs in qwenvl_timed 
                if max(start_time, qs['start']) < min(end_time, qs['end'])
            ]
            
            detail = {
                "whisper_segment": ws,
                "overlapping_qwenvl": overlapping_qwenvl,
                "decision": "whisper", # Default to Whisper
                "similarity": 0.0,
                "best_match_qwenvl": None,
                "all_similarities": [],
                "replaced_text": None
            }
            
            if overlapping_qwenvl:
                # Compute similarity between each overlapping QwenVL subtitle and the Whisper segment
                similarities = []
                for qs in overlapping_qwenvl:
                    similarity = fuzz.ratio(ws['text'], qs['text'])
                    similarities.append({
                        'qwenvl_text': qs['text'],
                        'similarity': similarity,
                        'qwenvl_segment': qs
                    })
                
                # 选择相似度最高的QwenVL字幕
                best_match = max(similarities, key=lambda x: x['similarity'])
                detail["similarity"] = best_match['similarity']
                detail["best_match_qwenvl"] = best_match['qwenvl_text']
                detail["all_similarities"] = similarities
                
                # 如果最高相似度超过阈值，则进行替换
                if best_match['similarity'] >= self.similarity_threshold:
                    detail["decision"] = "qwenvl"
                    detail["replaced_text"] = best_match['qwenvl_text']
                    
                    # 使用最匹配的QwenVL文本，但保留Whisper的时间戳
                    final_segments.append({
                        'start': ws['start'],
                        'end': ws['end'],
                        'text': best_match['qwenvl_text']
                    })
                else:
                    # 相似度不足，保留Whisper结果
                    final_segments.append(ws)
            else:
                # 没有重叠的QwenVL字幕，保留Whisper结果
                final_segments.append(ws)
            
            alignment_details.append(detail)
            whisper_idx += 1
            
        # 拼接最终文本
        final_text = " ".join([s['text'].strip() for s in final_segments])
        
        self.logger.info("融合完成")
        return final_text, alignment_details

    
    def get_ground_truth(self, video_name: str) -> str:
        """
        获取视频的标准答案（ground truth）
        从字幕文件中提取
        """
        # 尝试多个可能的字幕目录路径
        possible_subtitle_dirs = [
            Path("extracted_data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("../data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("speech2text/../data/闪婚幸运草的命中注定/带角色标注的字幕")
        ]
        
        for subtitle_dir in possible_subtitle_dirs:
            if subtitle_dir.exists():
                # 查找对应的字幕文件
                subtitle_file = subtitle_dir / f"{video_name}新.srt"
                if subtitle_file.exists():
                    return self._parse_srt_file(str(subtitle_file))
        
        self.logger.warning(f"未找到视频 {video_name} 的标准字幕文件")
        return ""
    
    def _parse_srt_file(self, srt_path: str) -> str:
        """解析SRT字幕文件，提取所有文本内容"""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的SRT解析：提取所有文本行（跳过时间戳和序号）
            lines = content.strip().split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                # 跳过序号行
                if line.isdigit():
                    continue
                # 跳过时间戳行 (格式: 00:00:00,000 --> 00:00:00,000)
                if '-->' in line:
                    continue
                # 跳过空行
                if not line:
                    continue
                
                # 去除发言人编号（如 "1 文本内容" -> "文本内容"）
                line = re.sub(r'^\d+\s+', '', line)
                text_lines.append(line)
            
            ground_truth = " ".join(text_lines)
            self.logger.debug(f"解析字幕文件: {len(text_lines)} 行文本")
            return ground_truth
            
        except Exception as e:
            self.logger.error(f"解析SRT文件失败 {srt_path}: {e}")
            return ""
    
    def calculate_metrics(self, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """
        计算四个评估指标（参考corpus_level_metrics_corrector.py）
        
        Args:
            predicted_text: 预测文本
            reference_text: 参考文本（标准答案）
            
        Returns:
            包含四个指标的字典
        """
        if not predicted_text or not reference_text:
            return {
                'bleu_score': 0.0,
                'chrf_score': 0.0,
                'cer': 1.0,  # 完全错误
                'character_accuracy': 0.0,
                'composite_score': 0.0
            }
        
        # 标准化文本处理
        def normalize_text(text: str) -> str:
            """标准化文本：去除首尾空白，规范化内部空格"""
            # 去除首尾空白
            text = text.strip()
            # 将多个连续空格替换为单个空格
            text = re.sub(r'\s+', ' ', text)
            # 去除标点符号前后的多余空格
            text = re.sub(r'\s*([，。！？；：])\s*', r'\1', text)
            return text
        
        # 清理和标准化文本
        pred_clean = normalize_text(predicted_text)
        ref_clean = normalize_text(reference_text)
        
        # 1. BLEU Score (使用sacrebleu)
        bleu = sacrebleu.corpus_bleu([pred_clean], [[ref_clean]], tokenize='zh')
        bleu_score = bleu.score
        
        # 2. chrF++ Score (使用sacrebleu)
        chrf = sacrebleu.corpus_chrf([pred_clean], [[ref_clean]], word_order=2)
        chrf_score = chrf.score
        
        # 3. CER (Character Error Rate) - 使用jiwer
        cer = jiwer.cer(ref_clean, pred_clean)
        
        # 4. Character Accuracy - 使用jiwer获取字符级对齐
        if ref_clean:
            output = jiwer.process_characters(ref_clean, pred_clean)
            correct_chars = output.hits
            total_chars_in_ref = output.hits + output.substitutions + output.deletions
            character_accuracy = correct_chars / total_chars_in_ref if total_chars_in_ref > 0 else 0.0
        else:
            character_accuracy = 1.0 if not pred_clean else 0.0
        
        # 5. 综合指标计算 (归一化后的四个指标平均值)
        bleu_normalized = bleu_score / 100.0
        chrf_normalized = chrf_score / 100.0
        cer_accuracy = 1.0 - cer  # CER转换为准确率
        
        # 综合指标 = (BLEU_norm + chrF++_norm + (1-CER) + CharAcc) / 4
        composite_score = (bleu_normalized + chrf_normalized + cer_accuracy + character_accuracy) / 4.0
        
        return {
            'bleu_score': bleu_score,
            'chrf_score': chrf_score,
            'cer': cer,
            'character_accuracy': character_accuracy,
            'composite_score': composite_score,
            'bleu_normalized': bleu_normalized,
            'chrf_normalized': chrf_normalized,
            'cer_accuracy': cer_accuracy
        }
    
    def process_video(self, video_path: str) -> FusionResult:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            FusionResult: 融合识别结果
        """
        start_time = time.time()
        video_file = Path(video_path).name
        video_name = Path(video_path).stem
        
        self.logger.info(f"开始处理视频: {video_file}")
        
        try:
            # 1. 获取视频信息
            duration, fps, frame_count = self.get_video_info(video_path)
            self.logger.info(f"视频信息: 时长{duration:.1f}s, {fps:.1f}fps, {frame_count}帧")
            
            # 2. 提取视频帧（使用缓存）
            frame_infos = self.extract_frames_uniform(video_path, video_name)
            
            # 3. QwenVL提取字幕（使用缓存）
            subtitle_segments = self.extract_subtitles_with_qwenvl(frame_infos, video_name)
            
            # 4. 字幕去重合并 (用于参考和评估)
            qwenvl_merged_text = self.deduplicate_adjacent_subtitles(subtitle_segments)
            
            # 5. Whisper语音识别 (带时间戳, 支持缓存)
            whisper_result = self.transcribe_with_whisper(video_path, video_name)
            whisper_original_text = whisper_result["text"].strip()
            whisper_segments = whisper_result["segments"]
            
            # 6. 核心融合逻辑
            # 使用自适应阈值逻辑完成 ASR/OCR 融合
            final_result, alignment_details = self.align_and_fuse_adaptive(
                subtitle_segments, whisper_segments
            )
            
            # 7. 获取标准答案并计算所有指标
            ground_truth = self.get_ground_truth(video_name)
            
            # 计算QwenVL原始结果指标
            qwenvl_metrics = self.calculate_metrics(qwenvl_merged_text, ground_truth) if ground_truth else {}
            
            # 计算Whisper原始结果指标
            whisper_original_metrics = self.calculate_metrics(whisper_original_text, ground_truth) if ground_truth else {}
            
            # 计算最终结果（融合后）指标
            final_metrics = self.calculate_metrics(final_result, ground_truth) if ground_truth else {}
            
            processing_time = time.time() - start_time
            
            # 统计替换信息
            replacements = [d for d in alignment_details if d['decision'] == 'qwenvl']
            stats = {
                "total_whisper_segments": len(whisper_segments),
                "qwenvl_replacements": len(replacements),
                "replacement_rate": len(replacements) / len(whisper_segments) if whisper_segments else 0,
                "average_similarity_of_replacements": sum(d['similarity'] for d in replacements) / len(replacements) if replacements else 0
            }

            # 创建结果对象
            result = FusionResult(
                video_file=video_file,
                total_duration=duration,
                alignment_details=alignment_details,
                qwenvl_merged_text=qwenvl_merged_text,
                whisper_original=whisper_original_text,
                final_result=final_result,
                ground_truth=ground_truth,
                processing_time=processing_time,
                stats=stats,
                
                # QwenVL原始结果指标
                qwenvl_bleu_score=qwenvl_metrics.get('bleu_score', 0.0),
                qwenvl_chrf_score=qwenvl_metrics.get('chrf_score', 0.0),
                qwenvl_cer=qwenvl_metrics.get('cer', 0.0),
                qwenvl_character_accuracy=qwenvl_metrics.get('character_accuracy', 0.0),
                qwenvl_composite_score=qwenvl_metrics.get('composite_score', 0.0),
                
                # Whisper原始结果指标
                whisper_original_bleu_score=whisper_original_metrics.get('bleu_score', 0.0),
                whisper_original_chrf_score=whisper_original_metrics.get('chrf_score', 0.0),
                whisper_original_cer=whisper_original_metrics.get('cer', 0.0),
                whisper_original_character_accuracy=whisper_original_metrics.get('character_accuracy', 0.0),
                whisper_original_composite_score=whisper_original_metrics.get('composite_score', 0.0),
                
                # 最终结果（融合）指标
                final_bleu_score=final_metrics.get('bleu_score', 0.0),
                final_chrf_score=final_metrics.get('chrf_score', 0.0),
                final_cer=final_metrics.get('cer', 0.0),
                final_character_accuracy=final_metrics.get('character_accuracy', 0.0),
                final_composite_score=final_metrics.get('composite_score', 0.0),
                
                metadata={
                    "frame_count": len(frame_infos),
                    "subtitle_segments": len(subtitle_segments),
                    "frame_interval": self.frame_interval,
                    "whisper_model": self.whisper_model_name,
                    "has_ground_truth": bool(ground_truth),
                    "used_frame_cache": self._check_cache_exists(video_name),
                    "used_qwenvl_cache": self._check_qwenvl_cache_exists(video_name),
                    "used_whisper_cache": self._check_whisper_cache_exists(video_name),
                    "fusion_params": {
                        "similarity_threshold": self.similarity_threshold,
                        "time_tolerance": self.time_tolerance
                    },
                    "qwenvl_metrics_details": qwenvl_metrics,
                    "whisper_original_metrics_details": whisper_original_metrics,
                    "final_metrics_details": final_metrics
                }
            )
            
            self.logger.info(f"视频处理完成: {video_file}, 耗时: {processing_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"处理视频 {video_file} 时出错: {e}")
            # 返回错误结果
            return FusionResult(
                video_file=video_file,
                total_duration=0.0,
                alignment_details=[],
                qwenvl_merged_text="",
                whisper_original="",
                final_result=f"处理失败: {str(e)}",
                ground_truth="",
                processing_time=time.time() - start_time,
                stats={},
                # 所有指标都设为默认错误值
                qwenvl_bleu_score=0.0,
                qwenvl_chrf_score=0.0,
                qwenvl_cer=1.0,
                qwenvl_character_accuracy=0.0,
                qwenvl_composite_score=0.0,
                whisper_original_bleu_score=0.0,
                whisper_original_chrf_score=0.0,
                whisper_original_cer=1.0,
                whisper_original_character_accuracy=0.0,
                whisper_original_composite_score=0.0,
                final_bleu_score=0.0,
                final_chrf_score=0.0,
                final_cer=1.0,
                final_character_accuracy=0.0,
                final_composite_score=0.0,
                metadata={"error": str(e)}
            )
    
    def process_multiple_videos(self, video_paths: List[str]) -> List[FusionResult]:
        """
        批量处理多个视频文件
        
        Args:
            video_paths: 视频文件路径列表
            
        Returns:
            List[FusionResult]: 所有视频的融合识别结果
        """
        self.logger.info(f"开始批量处理 {len(video_paths)} 个视频文件")
        
        results = []
        for i, video_path in enumerate(video_paths, 1):
            self.logger.info(f"处理进度: {i}/{len(video_paths)} - {Path(video_path).name}")
            
            try:
                result = self.process_video(video_path)
                results.append(result)
                
                # 显示处理进度和简要结果
                if result.ground_truth:
                    self.logger.info(f"完成 {Path(video_path).name}: 综合指标 {result.final_composite_score:.4f}")
                else:
                    self.logger.info(f"完成 {Path(video_path).name}: 处理成功，无标准答案")
                    
            except Exception as e:
                self.logger.error(f"处理视频 {video_path} 失败: {e}")
                # 创建错误结果记录
                error_result = FusionResult(
                    video_file=Path(video_path).name,
                    total_duration=0.0,
                    qwenvl_subtitles=[],
                    qwenvl_merged_text="",
                    whisper_original="",
                    whisper_guided="",
                    final_result=f"处理失败: {str(e)}",
                    ground_truth="",
                    processing_time=0.0,
                    confidence_score=0.0,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
                
        self.logger.info(f"批量处理完成: {len(results)} 个结果")
        return results
    
    def calculate_corpus_level_metrics(self, results: List[FusionResult]) -> Dict[str, Any]:
        """
        计算语料库级别的指标（将所有文本拼接后统一计算）
        
        Args:
            results: 所有有效的处理结果
            
        Returns:
            包含语料库级别指标的字典
        """
        # 过滤有效结果（有标准答案且无错误的）
        valid_results = [r for r in results if r.ground_truth and not r.metadata.get("error")]
        
        if not valid_results:
            return {
                "corpus_metrics": {
                    "qwenvl": {"bleu": 0.0, "chrf": 0.0, "cer": 1.0, "char_acc": 0.0, "composite": 0.0},
                    "whisper_original": {"bleu": 0.0, "chrf": 0.0, "cer": 1.0, "char_acc": 0.0, "composite": 0.0},
                    "final": {"bleu": 0.0, "chrf": 0.0, "cer": 1.0, "char_acc": 0.0, "composite": 0.0}
                },
                "valid_count": 0,
                "message": "没有有效的评估结果"
            }
        
        # 拼接所有文本
        qwenvl_texts = [r.qwenvl_merged_text for r in valid_results]
        whisper_original_texts = [r.whisper_original for r in valid_results]
        final_texts = [r.final_result for r in valid_results]
        ground_truth_texts = [r.ground_truth for r in valid_results]
        
        # 合并成长文本（用空格连接）
        qwenvl_corpus = " ".join(qwenvl_texts).strip()
        whisper_original_corpus = " ".join(whisper_original_texts).strip()
        final_corpus = " ".join(final_texts).strip()
        ground_truth_corpus = " ".join(ground_truth_texts).strip()
        
        self.logger.info(f"计算语料库级别指标: {len(valid_results)} 个有效结果")
        self.logger.info(f"语料库总长度: 参考文本 {len(ground_truth_corpus)} 字符")
        
        # 计算各方法的语料库级别指标
        qwenvl_corpus_metrics = self.calculate_metrics(qwenvl_corpus, ground_truth_corpus)
        whisper_original_corpus_metrics = self.calculate_metrics(whisper_original_corpus, ground_truth_corpus)
        final_corpus_metrics = self.calculate_metrics(final_corpus, ground_truth_corpus)
        
        return {
            "corpus_metrics": {
                "qwenvl": {
                    "bleu": qwenvl_corpus_metrics.get("bleu_score", 0.0),
                    "chrf": qwenvl_corpus_metrics.get("chrf_score", 0.0),
                    "cer": qwenvl_corpus_metrics.get("cer", 1.0),
                    "char_acc": qwenvl_corpus_metrics.get("character_accuracy", 0.0),
                    "composite": qwenvl_corpus_metrics.get("composite_score", 0.0)
                },
                "whisper_original": {
                    "bleu": whisper_original_corpus_metrics.get("bleu_score", 0.0),
                    "chrf": whisper_original_corpus_metrics.get("chrf_score", 0.0),
                    "cer": whisper_original_corpus_metrics.get("cer", 1.0),
                    "char_acc": whisper_original_corpus_metrics.get("character_accuracy", 0.0),
                    "composite": whisper_original_corpus_metrics.get("composite_score", 0.0)
                },
                "final": {
                    "bleu": final_corpus_metrics.get("bleu_score", 0.0),
                    "chrf": final_corpus_metrics.get("chrf_score", 0.0),
                    "cer": final_corpus_metrics.get("cer", 1.0),
                    "char_acc": final_corpus_metrics.get("character_accuracy", 0.0),
                    "composite": final_corpus_metrics.get("composite_score", 0.0)
                }
            },
            "valid_count": len(valid_results),
            "corpus_stats": {
                "qwenvl_length": len(qwenvl_corpus),
                "whisper_original_length": len(whisper_original_corpus),
                "final_length": len(final_corpus),
                "ground_truth_length": len(ground_truth_corpus)
            }
        }

    def find_available_tasks(self, directory: str) -> List[Tuple[str, str]]:
        """
        查找可用的任务（同时有视频文件和对应字幕文件的）
        
        Args:
            directory: 搜索目录
            
        Returns:
            List of (video_path, subtitle_path) tuples
        """
        base_dir = Path(directory)
        if not base_dir.exists():
            self.logger.warning(f"目录不存在: {directory}")
            return []
        
        # 查找所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(base_dir.glob(f"**/*{ext}"))
        
        # 字幕目录
        subtitle_dir = base_dir / "带角色标注的字幕"
        if not subtitle_dir.exists():
            self.logger.warning(f"字幕目录不存在: {subtitle_dir}")
            return []
        
        available_tasks = []
        
        for video_file in video_files:
            video_name = video_file.stem
            subtitle_file = subtitle_dir / f"{video_name}新.srt"
            
            if subtitle_file.exists():
                available_tasks.append((str(video_file), str(subtitle_file)))
                self.logger.debug(f"找到可用任务: {video_name}")
            else:
                self.logger.debug(f"跳过任务 {video_name}: 找不到对应字幕文件 {subtitle_file}")
        
        self.logger.info(f"找到 {len(available_tasks)} 个可用任务")
        return available_tasks

    def generate_batch_statistics(self, results: List[FusionResult]) -> Dict[str, Any]:
        """
        生成批处理统计信息
        
        Args:
            results: 所有处理结果
            
        Returns:
            包含统计信息的字典
        """
        # 过滤有效结果（有标准答案的）
        valid_results = [r for r in results if r.ground_truth and not r.metadata.get("error")]
        
        if not valid_results:
            return {
                "total_videos": len(results),
                "valid_videos": 0,
                "error_videos": len([r for r in results if r.metadata.get("error")]),
                "message": "没有有效的评估结果"
            }
        
        # 计算各项指标的统计信息
        def calc_stats(values):
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": sum(values) / len(values),
                "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                "min": min(values),
                "max": max(values)
            }
        
        # QwenVL指标统计
        qwenvl_stats = {
            "bleu": calc_stats([r.qwenvl_bleu_score for r in valid_results]),
            "chrf": calc_stats([r.qwenvl_chrf_score for r in valid_results]),
            "cer": calc_stats([r.qwenvl_cer for r in valid_results]),
            "character_accuracy": calc_stats([r.qwenvl_character_accuracy for r in valid_results]),
            "composite": calc_stats([r.qwenvl_composite_score for r in valid_results])
        }
        
        # Whisper原始指标统计
        whisper_original_stats = {
            "bleu": calc_stats([r.whisper_original_bleu_score for r in valid_results]),
            "chrf": calc_stats([r.whisper_original_chrf_score for r in valid_results]),
            "cer": calc_stats([r.whisper_original_cer for r in valid_results]),
            "character_accuracy": calc_stats([r.whisper_original_character_accuracy for r in valid_results]),
            "composite": calc_stats([r.whisper_original_composite_score for r in valid_results])
        }
        
        # 最终结果指标统计
        final_stats = {
            "bleu": calc_stats([r.final_bleu_score for r in valid_results]),
            "chrf": calc_stats([r.final_chrf_score for r in valid_results]),
            "cer": calc_stats([r.final_cer for r in valid_results]),
            "character_accuracy": calc_stats([r.final_character_accuracy for r in valid_results]),
            "composite": calc_stats([r.final_composite_score for r in valid_results])
        }
        
        # 性能提升统计
        improvements = {
            "qwenvl_vs_whisper_original": [
                r.qwenvl_composite_score - r.whisper_original_composite_score 
                for r in valid_results
            ],
            "final_vs_whisper_original": [
                r.final_composite_score - r.whisper_original_composite_score 
                for r in valid_results
            ],
            "final_vs_qwenvl": [
                r.final_composite_score - r.qwenvl_composite_score 
                for r in valid_results
            ]
        }
        
        improvement_stats = {
            key: calc_stats(values) for key, values in improvements.items()
        }
        
        # 处理时间统计
        processing_times = [r.processing_time for r in valid_results]
        time_stats = calc_stats(processing_times)
        
        return {
            "total_videos": len(results),
            "valid_videos": len(valid_results),
            "error_videos": len([r for r in results if r.metadata.get("error")]),
            "qwenvl_stats": qwenvl_stats,
            "whisper_original_stats": whisper_original_stats,
            "final_stats": final_stats,
            "improvement_stats": improvement_stats,
            "processing_time_stats": time_stats,
            "total_processing_time": sum(processing_times),
            "best_performing_video": max(valid_results, key=lambda x: x.final_composite_score).video_file if valid_results else None,
            "worst_performing_video": min(valid_results, key=lambda x: x.final_composite_score).video_file if valid_results else None
        }

    def save_results(self, results: List[FusionResult], output_dir: str = "speech2text/fusion_results"):
        """保存融合结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成批处理统计信息
        batch_stats = self.generate_batch_statistics(results)
        
        # 计算语料库级别指标
        corpus_metrics = self.calculate_corpus_level_metrics(results)
        
        # 保存详细JSON结果
        json_file = output_path / f"fusion_results_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "batch_statistics": batch_stats,
            "corpus_level_metrics": corpus_metrics,
            "model_info": {
                "qwenvl_api": self.qwenvl_api_url,
                "whisper_model": self.whisper_model_name,
                "frame_interval": self.frame_interval,
                "similarity_threshold": self.similarity_threshold,
                "time_tolerance": self.time_tolerance
            },
            "results": [
                {
                    "video_file": r.video_file,
                    "total_duration": r.total_duration,
                    "qwenvl_merged_text": r.qwenvl_merged_text,
                    "whisper_original": r.whisper_original,
                    "final_result": r.final_result,
                    "ground_truth": r.ground_truth,
                    "processing_time": r.processing_time,
                    "stats": r.stats,
                    
                    # QwenVL原始结果指标
                    "qwenvl_bleu_score": r.qwenvl_bleu_score,
                    "qwenvl_chrf_score": r.qwenvl_chrf_score,
                    "qwenvl_cer": r.qwenvl_cer,
                    "qwenvl_character_accuracy": r.qwenvl_character_accuracy,
                    "qwenvl_composite_score": r.qwenvl_composite_score,
                    
                    # Whisper原始结果指标
                    "whisper_original_bleu_score": r.whisper_original_bleu_score,
                    "whisper_original_chrf_score": r.whisper_original_chrf_score,
                    "whisper_original_cer": r.whisper_original_cer,
                    "whisper_original_character_accuracy": r.whisper_original_character_accuracy,
                    "whisper_original_composite_score": r.whisper_original_composite_score,
                    
                    # 最终结果指标
                    "final_bleu_score": r.final_bleu_score,
                    "final_chrf_score": r.final_chrf_score,
                    "final_cer": r.final_cer,
                    "final_character_accuracy": r.final_character_accuracy,
                    "final_composite_score": r.final_composite_score,
                    
                    "metadata": r.metadata,
                    # "alignment_details": r.alignment_details # 默认不保存过于详细的对齐信息
                }
                for r in results
            ]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 保存批处理统计摘要
        stats_file = output_path / f"batch_statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)
        
        # 保存详细的任务级别CSV结果
        tasks_csv_file = output_path / f"task_details_{timestamp}.csv"
        
        task_data = []
        for r in results:
            task_data.append({
                "task_name": Path(r.video_file).stem,
                "video_file": r.video_file,
                "duration": r.total_duration,
                "processing_time": r.processing_time,
                "has_ground_truth": bool(r.ground_truth),
                "has_error": bool(r.metadata.get("error")),
                
                # 文本输出
                "qwenvl_text": r.qwenvl_merged_text,
                "whisper_original": r.whisper_original,
                "final_result": r.final_result,
                "ground_truth": r.ground_truth,
                
                # 融合统计
                "total_whisper_segments": r.stats.get("total_whisper_segments"),
                "qwenvl_replacements": r.stats.get("qwenvl_replacements"),
                "replacement_rate": r.stats.get("replacement_rate"),
                "avg_replacement_similarity": r.stats.get("average_similarity_of_replacements"),

                # 任务级别指标
                "task_qwenvl_bleu": r.qwenvl_bleu_score,
                "task_qwenvl_chrf": r.qwenvl_chrf_score,
                "task_qwenvl_cer": r.qwenvl_cer,
                "task_qwenvl_char_acc": r.qwenvl_character_accuracy,
                "task_qwenvl_composite": r.qwenvl_composite_score,
                
                "task_whisper_original_bleu": r.whisper_original_bleu_score,
                "task_whisper_original_chrf": r.whisper_original_chrf_score,
                "task_whisper_original_cer": r.whisper_original_cer,
                "task_whisper_original_char_acc": r.whisper_original_character_accuracy,
                "task_whisper_original_composite": r.whisper_original_composite_score,
                
                "task_final_bleu": r.final_bleu_score,
                "task_final_chrf": r.final_chrf_score,
                "task_final_cer": r.final_cer,
                "task_final_char_acc": r.final_character_accuracy,
                "task_final_composite": r.final_composite_score,
                
                # 技术细节
                "frame_count": r.metadata.get("frame_count", 0),
                "subtitle_segments": r.metadata.get("subtitle_segments", 0),
                "used_frame_cache": r.metadata.get("used_frame_cache", False),
                "used_qwenvl_cache": r.metadata.get("used_qwenvl_cache", False),
                "used_whisper_cache": r.metadata.get("used_whisper_cache", False),
                "error_message": r.metadata.get("error", "")
            })
        
        task_df = pd.DataFrame(task_data)
        # 按任务名称排序
        task_df = task_df.sort_values('task_name')
        task_df.to_csv(tasks_csv_file, index=False, encoding='utf-8')
        
        # 新增：为每个任务保存详细的对齐/替换日志
        alignment_logs_dir = output_path / f"alignment_logs_{timestamp}"
        alignment_logs_dir.mkdir(exist_ok=True)
        
        for r in results:
            if not r.alignment_details:
                continue
            
            log_file = alignment_logs_dir / f"{Path(r.video_file).stem}_alignment.csv"
            log_data = []
            for detail in r.alignment_details:
                ws = detail['whisper_segment']
                qwenvl_texts = [qs['text'] for qs in detail['overlapping_qwenvl']]
                
                # 记录所有相似度
                all_sims = detail.get('all_similarities', [])
                sim_details = " | ".join([f"{s['qwenvl_text']}({s['similarity']:.1f})" for s in all_sims])
                
                log_data.append({
                    "whisper_start": ws['start'],
                    "whisper_end": ws['end'],
                    "whisper_text": ws['text'],
                    "overlapping_qwenvl_count": len(detail['overlapping_qwenvl']),
                    "all_qwenvl_similarities": sim_details,
                    "best_similarity": detail['similarity'],
                    "best_match_qwenvl": detail.get('best_match_qwenvl', ''),
                    "decision": detail['decision'],
                    "final_text": detail['replaced_text'] if detail['decision'] == 'qwenvl' else ws['text']
                })
            
            log_df = pd.DataFrame(log_data)
            log_df.to_csv(log_file, index=False, encoding='utf-8')

        # 保存CSV结果（包含评估指标）
        csv_file = output_path / f"fusion_summary_{timestamp}.csv"
        
        csv_data = []
        for r in results:
            csv_data.append({
                "video_file": r.video_file,
                "duration": r.total_duration,
                "final_result": r.final_result,
                "ground_truth": r.ground_truth,
                "qwenvl_text": r.qwenvl_merged_text,
                "whisper_original": r.whisper_original,
                "processing_time": r.processing_time,
                
                # 融合统计
                "replacement_rate": r.stats.get("replacement_rate"),
                "avg_replacement_similarity": r.stats.get("average_similarity_of_replacements"),

                # QwenVL原始结果指标
                "qwenvl_bleu_score": r.qwenvl_bleu_score,
                "qwenvl_chrf_score": r.qwenvl_chrf_score,
                "qwenvl_cer": r.qwenvl_cer,
                "qwenvl_character_accuracy": r.qwenvl_character_accuracy,
                "qwenvl_composite_score": r.qwenvl_composite_score,
                
                # Whisper原始结果指标
                "whisper_original_bleu_score": r.whisper_original_bleu_score,
                "whisper_original_chrf_score": r.whisper_original_chrf_score,
                "whisper_original_cer": r.whisper_original_cer,
                "whisper_original_character_accuracy": r.whisper_original_character_accuracy,
                "whisper_original_composite_score": r.whisper_original_composite_score,
                
                # 最终结果指标（融合）
                "final_bleu_score": r.final_bleu_score,
                "final_chrf_score": r.final_chrf_score,
                "final_cer": r.final_cer,
                "final_character_accuracy": r.final_character_accuracy,
                "final_composite_score": r.final_composite_score,
                
                # 其他信息
                "frame_count": r.metadata.get("frame_count", 0),
                "subtitle_segments": r.metadata.get("subtitle_segments", 0),
                "has_ground_truth": r.metadata.get("has_ground_truth", False)
            })
        
        df = pd.DataFrame(csv_data)
        # 按最终综合指标降序排序
        if 'final_composite_score' in df.columns:
            df = df.sort_values('final_composite_score', ascending=False)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"结果已保存:")
        self.logger.info(f"  详细结果: {json_file}")
        self.logger.info(f"  统计摘要: {stats_file}")
        self.logger.info(f"  任务详情: {tasks_csv_file}")
        self.logger.info(f"  汇总表格: {csv_file}")
        self.logger.info(f"  对齐日志 (CSV): {alignment_logs_dir}")
        
        return batch_stats, corpus_metrics

def find_video_files(directory: str, extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']) -> List[str]:
    """
    在指定目录中查找视频文件
    
    Args:
        directory: 搜索目录
        extensions: 支持的视频格式扩展名
        
    Returns:
        视频文件路径列表
    """
    video_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        return video_files
    
    for ext in extensions:
        video_files.extend(directory_path.glob(f"**/*{ext}"))
    
    return [str(f) for f in sorted(video_files)]

def display_batch_results(results: List[FusionResult], batch_stats: Dict[str, Any], corpus_metrics: Dict[str, Any]):
    """
    显示批处理结果摘要
    
    Args:
        results: 所有处理结果
        batch_stats: 批处理统计信息
        corpus_metrics: 语料库级别指标
    """
    print(f"\n{'='*80}")
    print("🎯 QwenVL + Whisper 批量融合增强结果")
    print(f"{'='*80}")
    
    # 基本统计
    print(f"📊 处理统计:")
    print(f"   总任务数: {batch_stats['total_videos']}")
    print(f"   有效评估: {batch_stats['valid_videos']}")
    print(f"   处理失败: {batch_stats['error_videos']}")
    
    if batch_stats['valid_videos'] > 0:
        print(f"   总处理时间: {batch_stats['total_processing_time']:.1f}s")
        print(f"   平均处理时间: {batch_stats['processing_time_stats']['mean']:.1f}s")
        
        # 语料库级别指标（最重要的）
        print(f"\n🎯 语料库级别最终指标 (所有文本拼接后计算):")
        corpus = corpus_metrics['corpus_metrics']
        print(f"   有效任务数: {corpus_metrics['valid_count']}")
        
        print(f"\n🔸 QwenVL (语料库级别):")
        qwenvl = corpus['qwenvl']
        print(f"   BLEU: {qwenvl['bleu']:.4f}")
        print(f"   chrF++: {qwenvl['chrf']:.4f}")
        print(f"   CER: {qwenvl['cer']:.4f}")
        print(f"   字符准确率: {qwenvl['char_acc']:.4f}")
        print(f"   🎯 综合指标: {qwenvl['composite']:.4f}")
        
        print(f"\n🔸 Whisper原始 (语料库级别):")
        whisper = corpus['whisper_original']
        print(f"   BLEU: {whisper['bleu']:.4f}")
        print(f"   chrF++: {whisper['chrf']:.4f}")
        print(f"   CER: {whisper['cer']:.4f}")
        print(f"   字符准确率: {whisper['char_acc']:.4f}")
        print(f"   🎯 综合指标: {whisper['composite']:.4f}")
        
        print(f"\n🔸 融合结果 (语料库级别):")
        final = corpus['final']
        print(f"   BLEU: {final['bleu']:.4f}")
        print(f"   chrF++: {final['chrf']:.4f}")
        print(f"   CER: {final['cer']:.4f}")
        print(f"   字符准确率: {final['char_acc']:.4f}")
        print(f"   🎯 综合指标: {final['composite']:.4f}")
        
        # 语料库级别的性能提升
        print(f"\n� 语料库级别性能提升:")
        qwenvl_vs_whisper = qwenvl['composite'] - whisper['composite']
        final_vs_whisper = final['composite'] - whisper['composite']
        final_vs_qwenvl = final['composite'] - qwenvl['composite']
        
        print(f"   QwenVL vs Whisper原始: {qwenvl_vs_whisper:+.4f}")
        print(f"   融合结果 vs Whisper原始: {final_vs_whisper:+.4f}")
        print(f"   融合结果 vs QwenVL: {final_vs_qwenvl:+.4f}")
        
        # 任务级别平均指标（作为参考）
        print(f"\n📊 任务级别平均指标 (仅供参考):")
        improvements = batch_stats['improvement_stats']
        print(f"   平均性能提升 (融合 vs Whisper): {improvements['final_vs_whisper_original']['mean']:+.4f} ± {improvements['final_vs_whisper_original']['std']:.4f}")
        
        # 最佳和最差表现
        if batch_stats['best_performing_video']:
            print(f"\n🏆 任务级别最佳表现: {batch_stats['best_performing_video']}")
            print(f"🔻 任务级别最差表现: {batch_stats['worst_performing_video']}")
        
        # 显示前5个和后5个结果
        valid_results = [r for r in results if r.ground_truth and not r.metadata.get("error")]
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x.final_composite_score, reverse=True)
            
            print(f"\n🏅 Top 5 任务表现:")
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"   {i}. {Path(result.video_file).stem}: {result.final_composite_score:.4f}")
            
            if len(sorted_results) > 5:
                print(f"\n🔻 Bottom 5 任务表现:")
                for i, result in enumerate(sorted_results[-5:], len(sorted_results)-4):
                    print(f"   {i}. {Path(result.video_file).stem}: {result.final_composite_score:.4f}")
    
    else:
        print(f"\n⚠️  没有有效的评估结果")

def main():
    """主函数 - 支持单个和批量处理"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="QwenVL + Whisper 融合增强系统")
    parser.add_argument("--video", type=str, 
                       help="单个视频文件路径")
    parser.add_argument("--video-dir", type=str,
                       help="视频目录路径，处理目录下所有视频文件")
    parser.add_argument("--video-pattern", type=str,
                       help="视频文件匹配模式，如 'data/*/*.mp4'")
    parser.add_argument("--qwenvl-api", type=str, 
                       default="http://localhost:8000/v1/chat/completions",
                       help="QwenVL API地址")
    parser.add_argument("--whisper-model", type=str, default="medium",
                       help="Whisper模型名称")
    parser.add_argument("--frame-interval", type=float, default=1.0,
                       help="帧提取间隔(秒)")
    parser.add_argument("--output-dir", type=str, 
                       default="speech2text/fusion_results",
                       help="结果输出目录")
    parser.add_argument("--clear-cache", action="store_true",
                       help="清理QwenVL缓存，强制重新处理")
    parser.add_argument("--clear-whisper-cache", action="store_true",
                          help="清理Whisper缓存，强制重新处理")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="最大处理视频数量限制")
    
    args = parser.parse_args()
    
    # 确定要处理的视频文件列表
    video_files = []
    
    if args.video:
        # 单个视频文件
        if Path(args.video).exists():
            video_files = [args.video]
        else:
            print(f"错误: 视频文件不存在: {args.video}")
            return
    elif args.video_dir:
        # 使用新的任务查找方法
        fusion_system = QwenVLWhisperFusion(
            qwenvl_api_url=args.qwenvl_api,
            whisper_model=args.whisper_model,
            frame_interval=args.frame_interval
        )
        available_tasks = fusion_system.find_available_tasks(args.video_dir)
        if not available_tasks:
            print(f"错误: 在目录 {args.video_dir} 中未找到可用任务（视频+字幕）")
            return
        video_files = [task[0] for task in available_tasks]  # 只取视频文件路径
    elif args.video_pattern:
        # 文件匹配模式
        video_files = glob.glob(args.video_pattern)
        if not video_files:
            print(f"错误: 没有匹配模式 {args.video_pattern} 的文件")
            return
    else:
        # 默认处理所有可用任务
        default_dir = "data/闪婚幸运草的命中注定"
        if Path(default_dir).exists():
            fusion_system = QwenVLWhisperFusion(
                qwenvl_api_url=args.qwenvl_api,
                whisper_model=args.whisper_model,
                frame_interval=args.frame_interval
            )
            available_tasks = fusion_system.find_available_tasks(default_dir)
            if available_tasks:
                video_files = [task[0] for task in available_tasks]
                print(f"默认处理目录 {default_dir} 中的所有可用任务")
            else:
                print(f"错误: 在默认目录 {default_dir} 中未找到可用任务")
                return
        else:
            print("错误: 请指定 --video, --video-dir 或 --video-pattern 参数，或确保默认目录存在")
            return
    
    # 限制处理数量
    if args.max_videos and len(video_files) > args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"限制处理视频数量为: {args.max_videos}")
    
    print(f"将处理 {len(video_files)} 个可用任务:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {Path(video_file).stem}")
    
    # 创建融合系统（如果还没创建）
    if 'fusion_system' not in locals():
        fusion_system = QwenVLWhisperFusion(
            qwenvl_api_url=args.qwenvl_api,
            whisper_model=args.whisper_model,
            frame_interval=args.frame_interval
        )
    
    # 如果需要清理缓存
    if args.clear_cache:
        for video_file in video_files:
            video_name = Path(video_file).stem
            fusion_system.clear_qwenvl_cache(video_name)
        print(f"已清理 {len(video_files)} 个视频的QwenVL缓存")

    if args.clear_whisper_cache:
        for video_file in video_files:
            video_name = Path(video_file).stem
            fusion_system.clear_whisper_cache(video_name)
        print(f"已清理 {len(video_files)} 个视频的Whisper缓存")
    
    # 处理视频（单个或批量）
    if len(video_files) == 1:
        # 单个视频处理
        result = fusion_system.process_video(video_files[0])
        results = [result]
        
        # 显示单个结果
        print(f"\n{'='*60}")
        print("🎯 QwenVL + Whisper 融合增强结果")
        print(f"{'='*60}")
        print(f"📹 视频文件: {result.video_file}")
        print(f"⏱️  处理时长: {result.processing_time:.1f}s")
        print(f"🎯 置信度: {result.confidence_score:.3f}")
        
        print(f"\n📝 QwenVL字幕:")
        print(f"   {result.qwenvl_merged_text}")
        print(f"\n🎤 Whisper原始:")
        print(f"   {result.whisper_original}")
        print(f"\n🎤 Whisper引导:")
        print(f"   {result.whisper_guided}")
        print(f"\n✨ 最终结果:")
        print(f"   {result.final_result}")
        
        # 如果有标准答案，显示评估指标
        if result.ground_truth:
            print(f"\n📚 标准答案:")
            print(f"   {result.ground_truth}")
            print(f"\n📊 详细评估指标对比:")
            print(f"{'='*60}")
            
            print(f"\n🔸 QwenVL原始结果指标:")
            print(f"   BLEU分数: {result.qwenvl_bleu_score:.4f}")
            print(f"   chrF++分数: {result.qwenvl_chrf_score:.4f}")
            print(f"   CER (字符错误率): {result.qwenvl_cer:.4f}")
            print(f"   字符准确率: {result.qwenvl_character_accuracy:.4f}")
            print(f"   🎯 综合指标: {result.qwenvl_composite_score:.4f}")
            
            print(f"\n🔸 Whisper原始结果指标:")
            print(f"   BLEU分数: {result.whisper_original_bleu_score:.4f}")
            print(f"   chrF++分数: {result.whisper_original_chrf_score:.4f}")
            print(f"   CER (字符错误率): {result.whisper_original_cer:.4f}")
            print(f"   字符准确率: {result.whisper_original_character_accuracy:.4f}")
            print(f"   🎯 综合指标: {result.whisper_original_composite_score:.4f}")
            
            print(f"\n🔸 最终结果指标 (Whisper引导):")
            print(f"   BLEU分数: {result.final_bleu_score:.4f}")
            print(f"   chrF++分数: {result.final_chrf_score:.4f}")
            print(f"   CER (字符错误率): {result.final_cer:.4f}")
            print(f"   字符准确率: {result.final_character_accuracy:.4f}")
            print(f"   🎯 综合指标: {result.final_composite_score:.4f}")
            
            # 显示性能提升情况
            qwenvl_vs_whisper_original = result.qwenvl_composite_score - result.whisper_original_composite_score
            final_vs_whisper_original = result.final_composite_score - result.whisper_original_composite_score
            final_vs_qwenvl = result.final_composite_score - result.qwenvl_composite_score
            
            print(f"\n📈 性能提升分析:")
            print(f"   QwenVL vs Whisper原始: {qwenvl_vs_whisper_original:+.4f}")
            print(f"   融合结果 vs Whisper原始: {final_vs_whisper_original:+.4f}")
            print(f"   融合结果 vs QwenVL: {final_vs_qwenvl:+.4f}")
        else:
            print(f"\n⚠️  未找到标准答案，无法计算评估指标")
    
    else:
        # 批量处理
        results = fusion_system.process_multiple_videos(video_files)
    
    # 保存结果并获取统计信息
    batch_stats, corpus_metrics = fusion_system.save_results(results, args.output_dir)
    
    # 如果是批量处理，显示统计摘要
    if len(video_files) > 1:
        display_batch_results(results, batch_stats, corpus_metrics)
    else:
        # 单个任务也显示语料库级别指标（就是自己）
        if results[0].ground_truth:
            corpus_metrics = fusion_system.calculate_corpus_level_metrics(results)
            print(f"\n📊 语料库级别指标 (单任务):")
            corpus = corpus_metrics['corpus_metrics']
            print(f"   融合结果 BLEU: {corpus['final']['bleu']:.4f}")
            print(f"   融合结果 综合指标: {corpus['final']['composite']:.4f}")

if __name__ == "__main__":
    main()
