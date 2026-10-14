#!/usr/bin/env python3
"""
Compute speaker cosine similarity for TTS systems using Resemblyzer.
Evaluates GPT-SoVITS, F5-TTS, and CosyVoice2 against reference audio.
EdgeTTS is excluded (uses fixed neural voice, no speaker cloning).

Output: baseline_results/speaker_similarity.json
"""

import json
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

ROOT      = Path(__file__).parent
REF_DIR   = ROOT / Path("data/tts_ref_audio")
BASE      = ROOT / Path("baseline_results")
OUT_FILE  = BASE / "speaker_similarity.json"

SYSTEMS = {
    "GPT-SoVITS v3": {
        "audio_dir": ROOT / Path("tts_results"),
        "pattern": lambda spk, sent: f"speaker{spk}_japanese_{sent}.wav",
        "speakers": [1, 2, 3, 4, 7, 8],
        "sentences": [0, 1, 2],
    },
    "F5-TTS": {
        "audio_dir": ROOT / Path("tts_results_f5tts"),
        "pattern": lambda spk, sent: f"speaker{spk}_japanese_{sent}.wav",
        "speakers": list(range(1, 13)),
        "sentences": [0, 1, 2],
    },
    "CosyVoice2-0.5B": {
        "audio_dir": BASE / "cosyvoice2",
        "pattern": lambda spk, sent: f"{spk}_japanese_{sent}.wav",
        "speakers": list(range(1, 13)),
        "sentences": [0, 1, 2],
    },
}

def get_ref_embedding(encoder, speaker_id):
    """Concatenate first few ref clips and compute a single embedding."""
    ref_dir = REF_DIR / str(speaker_id)
    wavs = sorted(ref_dir.glob("*.wav"))
    if not wavs:
        return None
    # Use up to 10 s of ref audio
    combined = []
    total = 0.0
    import soundfile as sf
    for w in wavs:
        y, sr = sf.read(str(w))
        combined.append(y)
        total += len(y) / sr
        if total >= 10.0:
            break
    audio = np.concatenate(combined).astype(np.float32)
    wav_pp = preprocess_wav(audio, source_sr=sr)
    return encoder.embed_utterance(wav_pp)


def main():
    encoder = VoiceEncoder()
    print("VoiceEncoder loaded.")

    results = {}

    for system_name, cfg in SYSTEMS.items():
        print(f"\n=== {system_name} ===")
        sims = []
        details = []

        # Cache ref embeddings
        ref_embs = {}
        for spk in cfg["speakers"]:
            emb = get_ref_embedding(encoder, spk)
            if emb is not None:
                ref_embs[spk] = emb

        for spk in cfg["speakers"]:
            if spk not in ref_embs:
                print(f"  Speaker {spk}: no ref audio, skipping")
                continue
            for sent in cfg["sentences"]:
                wav_path = cfg["audio_dir"] / cfg["pattern"](spk, sent)
                if not wav_path.exists():
                    print(f"  Missing: {wav_path.name}")
                    continue
                try:
                    wav_pp = preprocess_wav(str(wav_path))
                    gen_emb = encoder.embed_utterance(wav_pp)
                    ref_emb = ref_embs[spk]
                    sim = float(np.dot(ref_emb, gen_emb) /
                                (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb)))
                    sims.append(sim)
                    details.append({"speaker": spk, "sentence": sent, "similarity": sim})
                    print(f"  Speaker {spk} sent {sent}: sim={sim:.4f}")
                except Exception as e:
                    print(f"  Speaker {spk} sent {sent}: ERROR {e}")

        avg_sim = float(np.mean(sims)) if sims else 0.0
        std_sim = float(np.std(sims)) if sims else 0.0
        n_speakers = len(set(d["speaker"] for d in details))
        print(f"  => Mean sim: {avg_sim:.4f} ± {std_sim:.4f} (n={len(sims)}, {n_speakers} speakers)")
        results[system_name] = {
            "mean_similarity": avg_sim,
            "std_similarity": std_sim,
            "n_samples": len(sims),
            "n_speakers": n_speakers,
            "details": details,
        }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")

    print("\n=== Summary ===")
    for sys_name, r in results.items():
        print(f"{sys_name}: {r['mean_similarity']:.4f} ± {r['std_similarity']:.4f} "
              f"(n={r['n_samples']}, {r['n_speakers']} speakers)")


if __name__ == "__main__":
    main()
