import torch
from cosyvoice.api import CosyVoiceTTS
import torchaudio
import os
from pathlib import Path

# Initialize CosyVoice
# Using the pre-trained model for zero-shot inference
cosyvoice = CosyVoiceTTS('speech_tts/CosyVoice-300M')

def main():
    # Load test data same as other TTS experiments
    # For baseline, we just hardcode the few examples or read from a file if needed
    # Here we demonstrate with the standard test set logic if applicable, 
    # but for "baseline" quick replacement, we will use a demo set.
    
    # Actually, we should try to match the other experiment: 12 speakers.
    # But CosyVoice requires 3s prompt. 
    ref_audio_dir = Path("data/tts_ref_audio")
    output_dir = Path("baseline_results/cosyvoice")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target text (same as before)
    target_text = "これは私の履歴書です、どうぞご覧ください。"
    
    # Limit to first 5 speakers for speed as requested "replacement"
    speakers = sorted([d for d in ref_audio_dir.iterdir() if d.is_dir()])[:5]
    
    for spk_dir in speakers:
        print(f"Processing {spk_dir.name}...")
        
        # Find a wav file for prompt
        ref_wavs = list(spk_dir.glob("*.wav"))
        if not ref_wavs:
            continue
            
        prompt_speech_16k = torchaudio.load(ref_wavs[0])[0]
        
        # Run inference
        output = cosyvoice.inference_zero_shot(target_text, '日本語', prompt_speech_16k)
        
        # Save
        out_name = output_dir / f"spk_{spk_dir.name}.wav"
        torchaudio.save(out_name, output['tts_speech'], 22050)
        
    print("CosyVoice synthesis complete.")

if __name__ == "__main__":
    main()
