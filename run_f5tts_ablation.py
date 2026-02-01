import os
import sys
import torch
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import tomli
from importlib.resources import files
from omegaconf import OmegaConf
from hydra.utils import get_class
from cached_path import cached_path

# Add F5-TTS to path if needed, though it's installed as package
# Imports
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
)

# Reuse similarity logic if possible, or reimplement simplistically
# Attempting to load Resemblyzer for similarity
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    HAS_METRICS = True
    print("Resemblyzer loaded.")
except ImportError:
    print("Resemblyzer not found. Skipping similarity calculation.")
    HAS_METRICS = False

def concatenate_audio(wav_files, target_duration=10.0, sr=None):
    audio_segments = []
    total_duration = 0
    current_sr = sr
    
    for f in wav_files:
        y, s = sf.read(f)
        if current_sr is None:
            current_sr = s
        elif s != current_sr:
             # Resample if needed using librosa - simplified: only use matching sr
             continue
        
        audio_segments.append(y)
        total_duration += len(y) / current_sr
        if total_duration >= target_duration:
            break
            
    if not audio_segments:
        return None, None
        
    combined = np.concatenate(audio_segments)
    return combined, current_sr

def run_f5_ablation(data_dir, output_dir, model_name="F5TTS_Base"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model (Defaults from CLI)
    vocoder_name = "vocos"
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_file = str(cached_path(f"hf://SWivid/F5-TTS/{model_name}/model_1200000.safetensors"))
    vocab_file = "" # Use default
    
    print(f"Loading {model_name}...")
    ema_model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
    )
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=device)
    
    # Data Preparation
    base_dir = Path(data_dir) # extracted_data
    # Use Speaker 1
    speaker_dir = base_dir / "tts_ref_audio/1"
    wav_files = sorted(list(speaker_dir.glob("*.wav")))
    
    if not wav_files:
        print("No wav files found.")
        return

    # Prepare Short (3s) and Long (10s) Refs
    short_audio, sr = concatenate_audio(wav_files, target_duration=3.0)
    short_ref_path = output_dir / "ref_short.wav"
    sf.write(short_ref_path, short_audio, sr)
    
    long_audio, sr = concatenate_audio(wav_files, target_duration=10.0)
    long_ref_path = output_dir / "ref_long.wav"
    sf.write(long_ref_path, long_audio, sr)
    
    # Note: F5-TTS infers Ref Text from audio? Or needs text?
    # CLI accepts ref_text. If not provided, it expects something?
    # Current Utils: preprocess_ref_audio_text(ref_audio, ref_text)
    # If ref_text is provided, it uses it.
    # For concatenated audio, we should concatenate text too.
    # Simplified: Use "..." as dummy ref text if model supports ASR? 
    # F5-TTS usually requires ref text for alignment.
    # We will extract text from accompanying .txt files.
    
    def get_text_for_wavs(files_used, duration_limit):
        text = ""
        dur = 0
        for f in files_used:
            y, s = sf.read(f)
            # Find txt
            txt_p = f.with_suffix(".txt")
            if txt_p.exists():
                text += txt_p.read_text().strip() + " "
            dur += len(y)/s
            if dur >= duration_limit:
                break
        return text.strip()

    short_ref_text = get_text_for_wavs(wav_files, 3.0)
    long_ref_text = get_text_for_wavs(wav_files, 10.0)
    
    # Target Text (Japanese)
    target_text = "こんにちは、これはテストです。"
    
    results = []
    
    for condition, ref_path, ref_text_val in [("short", short_ref_path, short_ref_text), ("long", long_ref_path, long_ref_text)]:
        print(f"Running {condition} ablation...")
        
        # Preprocess
        final_ref_audio, final_ref_text = preprocess_ref_audio_text(str(ref_path), ref_text_val)
        
        audio, final_sr, spec = infer_process(
            final_ref_audio,
            final_ref_text,
            target_text,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device
        )
        
        out_name = output_dir / f"f5_{condition}.wav"
        sf.write(out_name, audio, final_sr)
        
        similarity = 0
        if HAS_METRICS:
            # Embeddings
            ref_emb = encoder.embed_utterance(preprocess_wav(ref_path))
            gen_emb = encoder.embed_utterance(preprocess_wav(out_name))
            similarity = np.dot(ref_emb, gen_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb))
            
        results.append({
            "condition": condition,
            "ref_length": "3s" if condition == "short" else "10s",
            "similarity": float(similarity)
        })
        
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("F5 Ablation Results:")
    print(results)

if __name__ == "__main__":
    run_f5_ablation("data", "ablation_f5_results")
