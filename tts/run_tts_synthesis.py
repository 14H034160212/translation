import os
import sys
import torch
from pathlib import Path
import soundfile as sf

# Add GPT-SoVITS paths
sys.path.append(os.path.abspath("GPT-SoVITS"))
sys.path.append(os.path.abspath("GPT-SoVITS/GPT_SoVITS"))

from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

def run_tts(gpt_path, sovits_path, ref_wav_path, ref_text, target_text, output_path, target_lang="Japanese"):
    # Target lang for get_tts_wav is 'Chinese', 'Japanese', 'English', 'Korean', 'Cantonese' (for v3/v2)
    # The keys rely on the dictionary in inference_webui.py
    
    # Consume the generator, passing language keys to avoid UnboundLocalError
    # We pass the languages we intend to use
    for _ in change_sovits_weights(sovits_path=sovits_path, prompt_language="Chinese", text_language=target_lang):
        pass
    
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_wav_path,
        prompt_text=ref_text,
        prompt_language="Chinese", # Ref audio is Chinese
        text=target_text,
        text_language=target_lang,
        top_p=1,
        temperature=1,
    )
    
    result_list = list(synthesis_result)
    if result_list:
        sampling_rate, audio_data = result_list[-1]
        sf.write(output_path, audio_data, sampling_rate)
        return True
    return False

if __name__ == "__main__":
    GPT_PATH = "GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt"
    SOVITS_PATH = "GPT-SoVITS/GPT_SoVITS/pretrained_models/s2Gv3.pth"
    
    # Load weights once for GPT (it's a standard function)
    change_gpt_weights(gpt_path=GPT_PATH)
    
    REF_BASE_DIR = Path("data/tts_ref_audio")
    OUTPUT_DIR = Path("tts_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Target test sentences
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
    
    speakers = [d for d in REF_BASE_DIR.iterdir() if d.is_dir() and d.name != "unknown"]
    
    for speaker_dir in speakers:
        speaker_id = speaker_dir.name
        print(f"--- Processing Speaker {speaker_id} ---")
        
        ref_wavs = list(speaker_dir.glob("*.wav"))
        if not ref_wavs: continue
        
        # Pick the best reference (e.g., largest file for better quality)
        ref_wav = max(ref_wavs, key=lambda p: p.stat().st_size)
        ref_text_path = ref_wav.with_suffix('.txt')
        if not ref_text_path.exists(): continue
        
        with open(ref_text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
            
        for lang in ["Chinese", "Japanese"]:
            for i, target_text in enumerate(TEST_SENTENCES[lang]):
                output_name = f"speaker{speaker_id}_{lang.lower()}_{i}.wav"
                output_path = OUTPUT_DIR / output_name
                print(f"Synthesizing: {output_name}")
                try:
                    success = run_tts(GPT_PATH, SOVITS_PATH, str(ref_wav), ref_text, target_text, str(output_path), lang)
                    if success:
                        print(f"Done: {output_name}")
                except Exception as e:
                    print(f"Error synthesizing {output_name}: {e}")
