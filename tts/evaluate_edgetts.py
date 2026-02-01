
import os
import whisper
import argparse
import pandas as pd
from jiwer import wer
from pathlib import Path
from tqdm import tqdm

# Reference text matching the EdgeTTS generation order in baseline_tts_edgetts.py
# The script generated 1.mp3 to 12.mp3 corresponding to test_sentences
TEST_SENTENCES = [
    # Chinese (1-3)
    "这是我的个人简历，请您过目。",
    "宋氏集团的面试对我来说非常重要。",
    "我一定会查清楚母亲去世的真相。",
    # Japanese (4-6) -> mapped to 4,5,6
    "これは私の履歴書です、どうぞご覧ください。",
    "宋氏グループの面接は私にとって非常に重要です。",
    "母の死の真相を必ず突き止めます。",
    # English (7-9) -> mapped to 7,8,9
    "This is my resume, please have a look.",
    "The interview at Song Group is very important to me.",
    "I will definitely find out the truth about my mother's death.",
    # Korean (10-12) -> mapped to 10,11,12
    "이것은 제 이력서입니다, 검토 부탁드립니다.",
    "송씨 그룹의 면접은 저에게 매우 중요합니다.",
    "저는 반드시 어머니의 죽음에 대한 진상을 밝혀낼 것입니다."
]

def evaluate_edgetts(results_dir):
    print(f"Loading Whisper model...")
    model = whisper.load_model("base")
    
    results = []
    results_dir = Path(results_dir)
    
    # We expect 1.mp3 to 12.mp3
    # Note: EdgeTTS script might have generated them 1-indexed based on the list iteration
    
    print(f"Evaluating EdgeTTS results in {results_dir}...")
    
    total_wer = 0
    total_cer = 0
    count = 0
    
    # Check what files exist
    files = sorted(list(results_dir.glob("*.mp3")), key=lambda x: int(x.stem) if x.stem.isdigit() else 999)
    
    for file_path in tqdm(files):
        if not file_path.stem.isdigit():
            continue
            
        index = int(file_path.stem) - 1 # 1-based to 0-based
        if index < 0 or index >= len(TEST_SENTENCES):
            print(f"Skipping {file_path.name}: Index {index} out of range")
            continue
            
        target_text = TEST_SENTENCES[index]
        
        # Determine language for Whisper
        lang = "zh"
        if 3 <= index <= 5: lang = "ja"
        elif 6 <= index <= 8: lang = "en"
        elif 9 <= index <= 11: lang = "ko"
        
        # Transcribe
        result = model.transcribe(str(file_path), language=lang)
        predicted_text = result['text'].strip()
        
        from jiwer import wer, cer
        error_wer = wer(target_text, predicted_text)
        error_cer = cer(target_text, predicted_text)
        
        results.append({
            "file": file_path.name,
            "target": target_text,
            "predicted": predicted_text,
            "wer": error_wer,
            "cer": error_cer
        })
        
        total_wer += error_wer
        total_cer += error_cer if 'total_cer' in locals() else error_cer # init below
        count += 1
        
    avg_wer = total_wer / count if count > 0 else 0
    avg_cer = total_cer / count if count > 0 else 0
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "eval_metrics.csv", index=False)
    print(f"Saved metrics to {results_dir}/eval_metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="baseline_results/edgetts")
    args = parser.parse_args()
    
    evaluate_edgetts(args.results_dir)
