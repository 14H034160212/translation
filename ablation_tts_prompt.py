import os
import sys
import torch
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
import jiwer
from tqdm import tqdm

# Add GPT-SoVITS paths
base_path = Path("/home/qbao775/translation")
sys.path.append(str(base_path / "GPT-SoVITS"))
sys.path.append(str(base_path / "GPT-SoVITS/GPT_SoVITS"))

from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# Whisper for evaluation
import whisper

def concatenate_audio(files, target_duration=10.0):
    """Concatenate audio files until target duration is reached."""
    combined_audio = []
    current_duration = 0.0
    sr = None
    
    for f in files:
        y, s = sf.read(f)
        if sr is None:
            sr = s
        elif s != sr:
            # Resample if needed using librosa (simplified here assuming same sr)
            continue
            
        combined_audio.append(y)
        current_duration += len(y) / sr
        if current_duration >= target_duration:
            break
            
    if not combined_audio:
        return None, None
        
    final_audio = np.concatenate(combined_audio)
    return final_audio, sr

def run_tts_ablation(ref_dir, output_dir):
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load TTS Models
    # Using the pretrained weights as set up in Experiment 2
    # Construct absolute paths
    base_dir = Path("/home/qbao775/translation")
    gpt_path = base_dir / "GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt"
    sovits_path = base_dir / "GPT-SoVITS/GPT_SoVITS/pretrained_models/s2Gv3.pth"
    
    change_gpt_weights(gpt_path=str(gpt_path))
    # change_sovits_weights is a generator, it might error on yield due to missing args, but globals should be set
    try:
        list(change_sovits_weights(sovits_path=str(sovits_path)))
    except Exception as e:
        print(f"Ignored expected error in change_sovits_weights: {e}")
    
    # Test Sentence (Japanese)
    target_text = "こんにちは、これはテストです。" # "Hello, this is a test."
    target_lang = "Japanese"
    
    # Select a speaker with enough files
    speaker_id = "1" # Folder 1 has 1040 files
    speaker_dir = ref_dir / speaker_id
    wav_files = sorted(list(speaker_dir.glob("*.wav")))
    
    if not wav_files:
        print(f"No wav files found for speaker {speaker_id}")
        return

    # Condition A: Short (Target 3-4s)
    # Concatenate until at least 3s
    short_audio, sr = concatenate_audio(wav_files, target_duration=3.0)
    # Check if actually > 3s, if not enough files, use what we have (might fail but best effort)
    # Actually concatenate_audio cuts off after target_duration? 
    # My implementation: break if current_duration >= target.
    # So it will be >= 3.0s.
    
    short_ref_path = output_dir / "temp_short_ref.wav"
    sf.write(short_ref_path, short_audio, sr)
    
    # Condition B: Long (Concatenate target 10s)
    long_audio, sr = concatenate_audio(wav_files, target_duration=10.0)
    long_ref_path = output_dir / "temp_long_ref.wav"
    sf.write(long_ref_path, long_audio, sr)
    
    # Ref Texts (Dummy text for zero-shot specific reference mode, 
    # but GPT-SoVITS ideally needs reference text. 
    # If we don't have transcripts for these specific wavs easily loaded, 
    # we might rely on the model's ability to handle potentially empty ref text or generic?
    # Actually GPT-SoVITS requires ref text.
    # We should look for corresponding .txt files!)
    
    # Short Ref Text
    short_ref_text = ""
    current_dur = 0
    for f in wav_files:
        y, s = sf.read(f)
        current_dur += len(y) / s
        
        txt_p = speaker_dir / f"{f.stem}.txt"
        if txt_p.exists():
            with open(txt_p, 'r') as tf:
                short_ref_text += tf.read().strip() + " "
        
        if current_dur >= 3.0:
            break
        
    # Long Ref Text?
    # Concatenating text is tricky if we don't know exactly which files were used.
    # I'll iterate again to get text.
    long_ref_text = ""
    current_dur = 0
    for f in wav_files:
        y, s = sf.read(f)
        current_dur += len(y) / s
        
        txt_p = speaker_dir / f"{f.stem}.txt"
        with open(txt_p, 'r') as tf:
            long_ref_text += tf.read().strip() + " "
            
        if current_dur >= 10.0:
            break
            
    print("Running Short Context Inference...")
    # Inference Short
    gen_short = get_tts_wav(
        ref_wav_path=str(short_ref_path),
        prompt_text=short_ref_text,
        prompt_language="Chinese", # Assuming source is Chinese
        text=target_text,
        text_language=target_lang
    )
    # gen_short is a generator yielding (sr, audio)
    result_short = list(gen_short)[0]
    sf.write(output_dir / "output_short.wav", result_short[1], result_short[0])
    
    print("Running Long Context Inference...")
    # Inference Long
    gen_long = get_tts_wav(
        ref_wav_path=str(long_ref_path),
        prompt_text=long_ref_text.strip(),
        prompt_language="Chinese",
        text=target_text,
        text_language=target_lang
    )
    result_long = list(gen_long)[0]
    sf.write(output_dir / "output_long.wav", result_long[1], result_long[0])
    
    # Evaluator
    print("Evaluating with Whisper...")
    model_whisper = whisper.load_model("base")
    
    def eval_wer(audio_path, ground_truth):
        result = model_whisper.transcribe(str(audio_path), language="ja")
        hyp = result["text"]
        return jiwer.wer(ground_truth, hyp), hyp
        
    wer_short, text_short = eval_wer(output_dir / "output_short.wav", target_text)
    wer_long, text_long = eval_wer(output_dir / "output_long.wav", target_text)
    
    print(f"Short Context (3s) | WER: {wer_short:.2f} | Output: {text_short}")
    print(f"Long Context (10s) | WER: {wer_long:.2f} | Output: {text_long}")
    
    results = {
        "short_wer": wer_short,
        "long_wer": wer_long,
        "short_text": text_short,
        "long_text": text_long
    }
    
    with open(output_dir / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Use absolute paths
    base_dir = Path("/home/qbao775/translation")
    run_tts_ablation(base_dir / "data/tts_ref_audio", base_dir / "ablation_tts_results")
