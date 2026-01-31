#!/usr/bin/env python3
"""
QwenVL Evaluator
QwenVL video subtitle recognition evaluation built on the unified framework.

Main capabilities:
1. Inherit the unified evaluation framework (BaseEvaluator)
2. Implement video subtitle matching logic
3. Parse SRT subtitle files
4. Support timestamp matching and frame alignment
5. Compatible with existing QwenVL testing flow
"""

import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from unified_evaluation_framework import BaseEvaluator


class QwenVLEvaluator(BaseEvaluator):
    """QwenVL video subtitle recognition evaluator."""
    
    def __init__(self, results_dir: str = None, evaluation_mode: str = "text_only", show_matched_text: bool = True):
        """
        Initialize the QwenVL evaluator.

        Args:
            results_dir: Path to the test results directory.
            evaluation_mode: Evaluation mode ("segmented" or "text_only").
            show_matched_text: Whether to print matched text in the terminal.
        """
        super().__init__(evaluation_mode, show_matched_text)
        
        if results_dir is None:
            results_dir = self.find_latest_results_dir()
        
        self.results_dir = Path(results_dir)
        
        # Try multiple possible subtitle directory paths
        possible_subtitle_dirs = [
            Path("extracted_data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("../../data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("../data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("data/闪婚幸运草的命中注定/带角色标注的字幕"),
            Path("qwenvl/../../data/闪婚幸运草的命中注定/带角色标注的字幕")
        ]
        
        self.subtitle_dir = None
        for subtitle_dir in possible_subtitle_dirs:
            if subtitle_dir.exists():
                self.subtitle_dir = subtitle_dir
                break
        
        if self.subtitle_dir is None:
            # Use the first path as default, even if it doesn't exist
            self.subtitle_dir = possible_subtitle_dirs[0]
        
        print(f"📂 使用结果目录: {self.results_dir}")
        
        # Data storage
        self.raw_data = []
        self.subtitles_by_video = {}

    def find_latest_results_dir(self) -> str:
        """Find the latest QwenVL results directory."""
        # Try multiple possible paths
        possible_patterns = [
            "baseline_results/qwenvl_baseline_*",
            "qwenvl/baseline_results/qwenvl_baseline_*",
            "speech2text/qwenvl/baseline_results/qwenvl_baseline_*",
            "../qwenvl/baseline_results/qwenvl_baseline_*"
        ]
        
        for pattern in possible_patterns:
            result_dirs = glob.glob(pattern)
            if result_dirs:
                latest_dir = max(result_dirs, key=os.path.getmtime)
                return latest_dir
        
        raise FileNotFoundError("未找到任何 QwenVL 结果目录")

    def load_reference_data(self) -> Dict[str, Any]:
        """Load reference data (subtitle files)."""
        print("📖 加载字幕参考数据...")
        
        if not self.subtitle_dir.exists():
            raise FileNotFoundError(f"字幕目录不存在: {self.subtitle_dir}")
        
        # Preload all subtitle files
        subtitle_files = list(self.subtitle_dir.glob("*.srt"))
        print(f"✅ 找到 {len(subtitle_files)} 个字幕文件")
        
        return {"subtitle_dir": self.subtitle_dir, "subtitle_count": len(subtitle_files)}

    def load_prediction_data(self) -> List[Dict]:
        """Load prediction results data."""
        print("📊 加载 QwenVL 预测结果...")
        
        # Try loading JSON format
        json_file = self.results_dir / "detailed_results.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        else:
            # Try loading CSV format
            csv_file = self.results_dir / "detailed_results.csv"
            if not csv_file.exists():
                raise FileNotFoundError(f"未找到结果文件: {csv_file}")
            df = pd.read_csv(csv_file)
            self.raw_data = df.to_dict('records')

        print(f"✅ 加载了 {len(self.raw_data)} 条 QwenVL 测试结果")
        return self.raw_data

    def load_subtitle_file(self, video_file: str) -> List[Dict]:
        """Load subtitle file for a specific video."""
        if video_file in self.subtitles_by_video:
            return self.subtitles_by_video[video_file]

        video_name = Path(video_file).stem
        possible_subtitle_names = [f"{video_name}新.srt", f"{video_name}.srt"]

        subtitle_segments = []
        for subtitle_name in possible_subtitle_names:
            subtitle_path = self.subtitle_dir / subtitle_name
            if subtitle_path.exists():
                subtitle_segments = self.parse_srt_file(subtitle_path)
                break

        self.subtitles_by_video[video_file] = subtitle_segments
        return subtitle_segments

    def parse_srt_file(self, srt_path: Path) -> List[Dict]:
        """Parse an SRT subtitle file."""
        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0])
                time_line = lines[1]
                start_str, end_str = time_line.split(' --> ')
                start_time = self._parse_timestamp(start_str)
                end_time = self._parse_timestamp(end_str)

                text_lines = lines[2:]
                full_text = ' '.join(text_lines)

                speaker = None
                text = full_text
                speaker_match = re.match(r'^(\d+)\s+(.+)$', full_text)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)

                segments.append({
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text.strip(),
                    'speaker': speaker
                })
            except (ValueError, IndexError):
                continue

        return segments

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse a timestamp into seconds."""
        time_part, ms_part = timestamp_str.split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        milliseconds = int(ms_part)
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def match_predictions_to_references(self) -> List[Dict]:
        """Match predictions to reference subtitles."""
        print("\n🔄 以字幕为基准匹配 QwenVL 预测结果...")

        matches = []
        videos_processed = set()

        for video_file in set(item['video_file'] for item in self.raw_data):
            subtitle_segments = self.load_subtitle_file(video_file)
            if not subtitle_segments:
                print(f"⚠️ 视频 {video_file} 没有找到字幕文件，跳过")
                continue

            videos_processed.add(video_file)
            video_predictions = [item for item in self.raw_data if item['video_file'] == video_file]

            for segment in subtitle_segments:
                best_match = self.find_best_prediction_for_subtitle(segment, video_predictions)
                if best_match:
                    matches.append({
                        'item_id': video_file,
                        'video_file': video_file,
                        'reference': segment['text'],
                        'prediction': best_match['predicted_text'],
                        'timestamp': best_match['timestamp'],
                        'subtitle_start': segment['start_time'],
                        'subtitle_end': segment['end_time']
                    })

        print(f"✅ 处理了 {len(videos_processed)} 个视频")
        print(f"✅ 匹配了 {len(matches)} 个字幕片段")
        return matches

    def find_best_prediction_for_subtitle(self, subtitle: Dict, predictions: List[Dict]) -> Optional[Dict]:
        """Find the best-matching prediction for a subtitle segment."""
        start_time = subtitle['start_time']
        end_time = subtitle['end_time']
        tolerance = 0.5  # Time tolerance

        candidates = []
        for pred in predictions:
            timestamp = pred['timestamp']
            if start_time - tolerance <= timestamp <= end_time + tolerance:
                center_time = (start_time + end_time) / 2
                distance = abs(timestamp - center_time)
                candidates.append((pred, distance))

        if not candidates:
            return None

        best_pred, _ = min(candidates, key=lambda x: x[1])
        return best_pred

    def get_full_text_for_item(self, item_id: str) -> Tuple[str, str]:
        """Get full predicted and reference text for a video."""
        video_file = item_id  # For QwenVL, item_id is the video_file
        
        # Get reference text (subtitles)
        subtitle_segments = self.load_subtitle_file(video_file)
        if not subtitle_segments:
            reference_text = ""
        else:
            sorted_segments = sorted(subtitle_segments, key=lambda x: x['start_time'])
            reference_parts = [self.metrics_calculator.clean_text(s['text']) for s in sorted_segments 
                             if self.metrics_calculator.clean_text(s['text'])]
            reference_text = ' '.join(reference_parts)

        # Get predicted text
        video_predictions = [item for item in self.raw_data if item['video_file'] == video_file]
        if not video_predictions:
            predicted_text = ""
        else:
            sorted_predictions = sorted(video_predictions, key=lambda x: x['timestamp'])
            predicted_parts = [self.metrics_calculator.clean_text(p['predicted_text']) for p in sorted_predictions 
                             if self.metrics_calculator.clean_text(p['predicted_text'])]
            predicted_text = ' '.join(predicted_parts)

        return predicted_text, reference_text

    def calculate_segmented_evaluation(self, matches: List[Dict]) -> List[Dict]:
        """Compute segmented evaluation results."""
        print("\n📊 计算逐句匹配指标...")
        
        # Group by video
        matches_by_video = {}
        for match in matches:
            video_file = match['video_file']
            matches_by_video.setdefault(video_file, []).append(match)

        video_metrics = []
        for video_file, video_matches in matches_by_video.items():
            if self.show_matched_text:
                print(f"\n📝 视频 {video_file} 的匹配片段:")
                for i, match in enumerate(video_matches[:3]):  # Show first 3 matches
                    print(f"  片段{i+1}: 参考='{match['reference']}' | 预测='{match['prediction']}'")
                if len(video_matches) > 3:
                    print(f"  ... (共 {len(video_matches)} 个匹配片段)")

            # Compute video-level metrics
            video_metric = self.calculate_metrics_for_item_segmented(video_matches)
            video_metric.update({
                'item_id': video_file,
                'video_file': video_file,
                'total_subtitles': len(self.subtitles_by_video.get(video_file, []))
            })
            
            video_metrics.append(video_metric)
            print(f"  ✅ {video_file}: CER={video_metric['cer']:.3f}, ACC={video_metric['character_accuracy']:.3f}, "
                  f"BLEU={video_metric['bleu_score']:.3f}, chrF++={video_metric['chrf_plus_plus']:.3f}")

        return video_metrics

    def calculate_text_only_evaluation(self) -> List[Dict]:
        """Compute full-text evaluation results."""
        print("\n📊 计算整体文本匹配指标...")
        
        video_files = list(set(item['video_file'] for item in self.raw_data))
        video_metrics = []
        
        for video_file in video_files:
            if not self.load_subtitle_file(video_file):
                print(f"⚠️ 视频 {video_file} 没有找到字幕文件，跳过")
                continue
            
            video_metric = self.calculate_metrics_for_item_text_only(video_file)
            
            # Add extra statistics
            subtitle_segments = self.load_subtitle_file(video_file)
            video_predictions = [item for item in self.raw_data if item['video_file'] == video_file]
            video_metric.update({
                'video_file': video_file,
                'total_subtitles': len(subtitle_segments),
                'total_predictions': len(video_predictions)
            })
            
            video_metrics.append(video_metric)
            print(f"  ✅ {video_file}: CER={video_metric['cer']:.3f}, ACC={video_metric['character_accuracy']:.3f}, "
                  f"BLEU={video_metric['bleu_score']:.3f}, chrF++={video_metric['chrf_plus_plus']:.3f}")

        return video_metrics

    def run_qwenvl_evaluation(self, output_filename: str = None) -> Tuple[Dict, List[Dict]]:
        """Run QwenVL evaluation and save results."""
        overall_metrics, item_metrics = self.run_evaluation()
        
        # Save results
        if output_filename is None:
            output_filename = f"qwenvl_unified_results_{self.evaluation_mode}.json"
        
        output_path = self.results_dir / output_filename
        self.save_results(overall_metrics, item_metrics, output_path)
        
        return overall_metrics, item_metrics


def main():
    """Main entry point (CLI-compatible)."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QwenVL 统一评估框架')
    parser.add_argument('--mode', choices=['segmented', 'text_only'], default='text_only',
                       help='评估模式: segmented (逐句匹配) 或 text_only (整体文本匹配)')
    parser.add_argument('--results-dir', type=str, help='结果目录路径（可选）')
    parser.add_argument('--show-text', action='store_true', default=True,
                       help='显示匹配的文本内容（默认开启）')
    parser.add_argument('--no-show-text', action='store_false', dest='show_text',
                       help='不显示匹配的文本内容')
    
    args = parser.parse_args()
    
    evaluator = QwenVLEvaluator(
        results_dir=args.results_dir,
        evaluation_mode=args.mode,
        show_matched_text=args.show_text
    )
    
    evaluator.run_qwenvl_evaluation()


if __name__ == "__main__":
    main()
