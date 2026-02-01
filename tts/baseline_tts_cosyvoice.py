import os
import torch
import torchaudio
from modelscope import snapshot_download
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

# Configuration
# This requires installing cosyvoice: pip install cosyvoice-runtime or from source
# And modelscope
OUTPUT_DIR = "baseline_results/cosyvoice"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading CosyVoice-300M...")
    # This will download the model to cache
    # "speech_tts/CosyVoice-300M" is the model id in modelscope
    try:
        cosyvoice = CosyVoice('speech_tts/CosyVoice-300M')
    except Exception as e:
        print(f"Failed to load CosyVoice: {e}")
        print("Please ensure 'modelscope' and 'cosyvoice' are installed.")
        return

    # Test Text
    target_text = "こんにちは、これはテストです。"
    
    # Reference Audio (Speaker 1)
    ref_audio_path = "data/tts_ref_audio/1/11_0.wav"
    
    if not os.path.exists(ref_audio_path):
        print(f"Ref audio not found: {ref_audio_path}")
        # Create a dummy or skip
        return

    print(f"Synthesizing: {target_text}")
    # zero_shot inference
    prompt_speech_16k = load_wav(ref_audio_path, 16000)
    
    output = cosyvoice.inference_zero_shot(target_text, 'This is a prompt text placeholder', prompt_speech_16k)
    
    # Save output
    # output is usually a generator or list of dicts with 'tts_speech'
    for i, item in enumerate(output):
        torchaudio.save(f"{OUTPUT_DIR}/cosyvoice_output_{i}.wav", item['tts_speech'], 22050)
        print(f"Saved {OUTPUT_DIR}/cosyvoice_output_{i}.wav")

if __name__ == "__main__":
    main()
