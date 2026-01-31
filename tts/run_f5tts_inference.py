import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from importlib.resources import files
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
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

def run_f5tts_inference():
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model Constants
    model_name = "F5TTS_Base"
    vocoder_name = "vocos"
    repo_name = "F5-TTS"
    ckpt_step = 1200000
    ckpt_type = "safetensors"

    # Config loading (simplified from infer_cli.py)
    # We assume standard installed package layout
    config_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    model_cfg = OmegaConf.load(config_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    # Download/Cache Checkpoint
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))
    vocab_file = "" # Use default (embedded or loaded internally if not specified, infer_cli allows empty)

    # Load Model & Vocoder
    print("Loading Vocoder...")
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=device)
    
    print(f"Loading {model_name}...")
    ema_model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
    )

    # Experiment Data Setup (Matched with Experiment 2)
    REF_BASE_DIR = Path("data/tts_ref_audio")
    OUTPUT_DIR = Path("tts_results_f5tts")
    OUTPUT_DIR.mkdir(exist_ok=True)

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
        if not ref_wavs:
            continue
        
        # Pick best reference (largest file)
        ref_wav_path = max(ref_wavs, key=lambda p: p.stat().st_size)
        
        # Check for transcript
        ref_text_path = ref_wav_path.with_suffix('.txt')
        if not ref_text_path.exists():
            print(f"Skipping {ref_wav_path}, no transcript found.")
            continue
            
        with open(ref_text_path, 'r', encoding='utf-8') as f:
            ref_text_raw = f.read().strip()
            
        print(f"Reference: {ref_wav_path} | {ref_text_raw}")
        
        # Preprocess Reference (needed for F5-TTS?)
        # utils_infer.preprocess_ref_audio_text just ensures text exists, and calls another func.
        # But we can call it.
        # Note: preprocess_ref_audio_text is more about cleaning text if needed.
        
        for lang in ["Chinese", "Japanese"]:
            for i, target_text in enumerate(TEST_SENTENCES[lang]):
                output_name = f"speaker{speaker_id}_{lang.lower()}_{i}.wav"
                output_path = OUTPUT_DIR / output_name
                print(f"Synthesizing: {output_name}")
                
                try:
                    # Run Inference
                    audio_segment, final_sample_rate, _ = infer_process(
                        str(ref_wav_path),
                        ref_text_raw,
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
                        device=device,
                    )
                    
                    sf.write(output_path, audio_segment, final_sample_rate)
                    print(f"Done: {output_name}")
                    
                except Exception as e:
                    print(f"Error synthesizing {output_name}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    run_f5tts_inference()
