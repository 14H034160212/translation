import os
import re
import subprocess
from pathlib import Path

def parse_srt(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        
        # Time line: 00:00:01,000 --> 00:00:04,000
        time_match = re.search(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', lines[1])
        if not time_match: continue
        
        start_str = time_match.group(1).replace(',', '.')
        end_str = time_match.group(2).replace(',', '.')
        
        text = " ".join(lines[2:])
        # Look for speaker ID like "1 text" or "2: text"
        speaker_match = re.match(r'^(\d+)\s+(.+)$', text)
        if speaker_match:
            speaker_id = speaker_match.group(1)
            clean_text = speaker_match.group(2)
        else:
            # Try colon as backup
            speaker_match = re.match(r'^(\d+)\s*[：:]\s*(.+)$', text)
            if speaker_match:
                speaker_id = speaker_match.group(1)
                clean_text = speaker_match.group(2)
            else:
                speaker_id = "unknown"
                clean_text = text
            
        segments.append({
            "start": start_str,
            "end": end_str,
            "speaker": speaker_id,
            "text": clean_text
        })
    return segments

def extract_audio_clip(video_path, start, end, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-to", end,
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)

def preprocess_all(video_dir, srt_dir, output_base_dir):
    video_dir = Path(video_dir)
    srt_dir = Path(srt_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    srt_files = list(srt_dir.glob("*.srt"))
    for srt_file in srt_files:
        video_name = srt_file.stem.replace("新", "")
        video_path = video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"Video not found for {srt_file.name}")
            continue
            
        print(f"Processing {srt_file.name}...")
        segments = parse_srt(srt_file)
        for i, seg in enumerate(segments):
            speaker_dir = output_base_dir / seg['speaker']
            speaker_dir.mkdir(exist_ok=True)
            
            output_path = speaker_dir / f"{video_name}_{i}.wav"
            extract_audio_clip(video_path, seg['start'], seg['end'], output_path)
            
            # Save transcript for fine-tuning/reference
            with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(seg['text'])

if __name__ == "__main__":
    VIDEO_DIR = "extracted_data/闪婚幸运草的命中注定"
    SRT_DIR = "extracted_data/闪婚幸运草的命中注定/带角色标注的字幕"
    OUTPUT_DIR = "data/tts_ref_audio"
    preprocess_all(VIDEO_DIR, SRT_DIR, OUTPUT_DIR)
