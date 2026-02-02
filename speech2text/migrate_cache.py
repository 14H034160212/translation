import json
from pathlib import Path
from datetime import datetime

def migrate_to_cache(json_path, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    by_video = {}
    for item in data:
        v_file = item['video_file']
        v_name = Path(v_file).stem
        if v_name not in by_video:
            by_video[v_name] = []
        by_video[v_name].append([item['timestamp'], item['predicted_text']])
        
    for v_name, subtitles in by_video.items():
        cache_file = cache_dir / f"{v_name}_qwenvl_results.json"
        
        # Sort by timestamp
        subtitles.sort(key=lambda x: x[0])
        
        cache_data = {
            "video_name": v_name,
            "timestamp": datetime.now().isoformat(),
            "subtitles": subtitles,
            "frame_interval": 0.2  # 5fps
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
    print(f"Migrated {len(by_video)} videos to {cache_dir}")

if __name__ == "__main__":
    migrate_to_cache(
        "baseline_results/qwen3vl_4b_fps5/qwen3vl_baseline_final/detailed_results.json",
        "speech2text/qwenvl_cache_hybrid_no_post_3090"
    )
