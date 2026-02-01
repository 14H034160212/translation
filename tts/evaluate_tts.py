import os
import whisper
from jiwer import wer
from pathlib import Path

def evaluate_tts(results_dir, test_sentences):
    model = whisper.load_model("base")
    results = []
    
    results_dir = Path(results_dir)
    audio_files = list(results_dir.glob("*.wav"))
    print(f"Evaluating {len(audio_files)} files...")
    
    # Map index to sentence
    # speaker{id}_{lang}_{index}.wav
    
    for audio_path in audio_files:
        name = audio_path.stem
        parts = name.split('_')
        if len(parts) < 3: continue
        
        lang_key = parts[1].capitalize() # chinese or japanese -> Chinese or Japanese
        index = int(parts[2])
        
        target_text = test_sentences[lang_key][index]
        
        print(f"Transcribing {audio_path.name}...")
        # Use task='transcribe' and language=...
        lang_code = 'zh' if lang_key == 'Chinese' else 'ja'
        result = model.transcribe(str(audio_path), language=lang_code)
        transcribed_text = result['text'].strip()
        
        # Calculate WER and CER
        from jiwer import wer, cer
        error_wer = wer(target_text, transcribed_text)
        error_cer = cer(target_text, transcribed_text)
        
        results.append({
            "name": audio_path.name,
            "target": target_text,
            "transcribed": transcribed_text,
            "wer": error_wer,
            "cer": error_cer
        })
        
        print(f"Target: {target_text}")
        print(f"Got   : {transcribed_text}")
        print(f"WER   : {error_wer:.4f}, CER: {error_cer:.4f}\n")
    
    # Calculate average metrics
    if results:
        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        print(f"--- Final Evaluation ---")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")

if __name__ == "__main__":
    TEST_SENTENCES = {
        "Chinese": [
            "这是我的个人简历，请您过目。",
            "宋氏集团的面试对我来说非常重要。",
            "我一定会查清楚母亲去世的真相。"
        ],
        "Japanese": [
            "これは私の履歴書です、どうぞご覧ください。",
            "宋氏グループの面接は私にとって非常に重要です。",
            "母の死の真相を必ず突き止めます。"
        ]
    }
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="tts_results", help="Directory containing TTS results")
    args = parser.parse_args()

    TEST_SENTENCES = {
        "Chinese": [
            "这是我的个人简历，请您过目。",
            "宋氏集团的面试对我来说非常重要。",
            "我一定会查清楚母亲去世的真相。"
        ],
        "Japanese": [
            "これは私の履歴書です、どうぞご覧ください。",
            "宋氏グループの面接は私にとって非常に重要です。",
            "母の死の真相を必ず突き止めます。"
        ]
    }
    
    evaluate_tts(args.results_dir, TEST_SENTENCES)
