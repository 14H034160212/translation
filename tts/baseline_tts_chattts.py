import os
import torch
import ChatTTS
import soundfile as sf
import torchaudio
from pathlib import Path

# Usage: python tts/baseline_tts_chattts.py

OUTPUT_DIR = "baseline_results/chattts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading ChatTTS...")
    chat = ChatTTS.Chat()
    # Download models if needed
    chat.load(compile=False) # compile=True for faster inference but requires triton/cuda setup

    # Load ChatTTS
    # We should iterate over data/tts_ref_audio
    ref_dir = Path("data/tts_ref_audio")
    speakers = sorted(list(ref_dir.glob("*")))
    
    target_text = ["こんにちは、これはテストです。", "ドラマの字幕翻訳をしています。"]
    
    for i, spk_dir in enumerate(speakers):
        if not spk_dir.is_dir(): continue
        
        # In a real scenario, we would refine the prompt based on speaker audio
        # ChatTTS supports zero-shot speaker conditioning if configured
        # For baseline, we just run inference to generate samples
        print(f"Synthesizing for speaker {spk_dir.name}")
        wavs = chat.infer(target_text)
        
        # Save one sample per speaker for comparison
        wav = wavs[0]
        if isinstance(wav, list):
             wav = torch.tensor(wav).unsqueeze(0)
             
        out_name = f"{OUTPUT_DIR}/spk_{spk_dir.name}_sample.wav"
        # wav is likely 1D numpy array if not list
        if isinstance(wav, torch.Tensor):
             tensor_wav = wav
        else:
             tensor_wav = torch.from_numpy(wav).unsqueeze(0)
             
        torchaudio.save(out_name, tensor_wav, 24000)
        print(f"Saved {out_name}")
        if i >= 2: break # Limit samples for baseline to save time
    
    for i, wav in enumerate(wavs):
        # Audio is usually (1, T) or list of numpy
        # Save to file
        # ChatTTS output sample rate is usually 24000
        output_path = f"{OUTPUT_DIR}/output_{i}.wav"
        # If wav is list or tensor
        if isinstance(wav, list):
             wav = torch.tensor(wav).unsqueeze(0)
             
        torchaudio.save(output_path, torch.from_numpy(wav[0]).unsqueeze(0), 24000)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
