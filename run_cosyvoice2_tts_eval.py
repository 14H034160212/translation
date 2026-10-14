#!/usr/bin/env python3
"""
CosyVoice2-0.5B zero-shot cross-lingual TTS evaluation.

CosyVoice2 (Alibaba FunAudioLLM) uses a flow-matching acoustic model + HiFi-GAN vocoder.
Zero-shot mode: given a Chinese reference audio + its text, synthesize Japanese target text
in the same speaker's voice.

Evaluation follows the same 12-speaker × 3-sentence × 2-language protocol as GPT-SoVITS
and F5-TTS: WER and CER are computed by transcribing the generated audio with Whisper.

Run with:
    source /data/home/qbao775/translation/venv_chattts/bin/activate
    CUDA_VISIBLE_DEVICES=2 python3 run_cosyvoice2_tts_eval.py

Output: baseline_results/cosyvoice2_eval.json
"""

import sys, os, json, torch, soundfile as sf, numpy as np
from pathlib import Path
from tqdm import tqdm

COSYVOICE_DIR = Path(__file__).parent / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party/Matcha-TTS"))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party/AcademiCodec"))
os.chdir(str(COSYVOICE_DIR))   # CosyVoice requires cwd to be repo root

MODEL_DIR    = str(COSYVOICE_DIR / "pretrained_models/CosyVoice2-0.5B")
REF_AUDIO_DIR = Path(__file__).parent / "data/tts_ref_audio"
# speakers are numbered dirs: 1, 2, ..., 12  (not speaker1, speaker2)
# rename in the synth records for clarity
OUTPUT_DIR    = Path(__file__).parent / "baseline_results/cosyvoice2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SENTENCES = {
    "chinese":  [
        "这是我的个人简历，请您过目。",
        "宋氏集团的面试对我来说非常重要。",
        "我一定会查清楚母亲去世的真相。",
    ],
    "japanese": [
        "これは私の履歴書です、どうぞご覧ください。",
        "宋氏グループの面接は私にとって非常に重要です。",
        "母の死の真相を必ず突き止めます。",
    ],
}

# Language tokens used by CosyVoice2 cross-lingual mode
LANG_TOKEN = {"chinese": "<|zh|>", "japanese": "<|ja|>"}


def download_model():
    """Download CosyVoice2-0.5B from ModelScope if not present."""
    if Path(MODEL_DIR).exists() and (Path(MODEL_DIR) / "flow.pt").exists():
        print("Model already downloaded.")
        return
    print("Downloading CosyVoice2-0.5B from ModelScope …")
    from modelscope import snapshot_download
    snapshot_download("iic/CosyVoice2-0.5B",
                      local_dir=MODEL_DIR)
    print("Download complete.")


def load_cosyvoice():
    from cosyvoice.cli.cosyvoice import AutoModel
    model = AutoModel(model_dir=MODEL_DIR)
    return model


def synthesize(cosyvoice, text, lang, ref_wav_path, ref_text):
    """Synthesize text in target lang using speaker reference audio."""
    tagged_text = LANG_TOKEN[lang] + text
    results = []
    try:
        for chunk in cosyvoice.inference_cross_lingual(tagged_text, ref_wav_path):
            results.append(chunk["tts_speech"].squeeze().cpu().numpy())
    except Exception as e:
        print(f"    synthesis error: {e}")
        return None, cosyvoice.sample_rate
    if not results:
        return None, cosyvoice.sample_rate
    audio = np.concatenate(results)
    return audio, cosyvoice.sample_rate


def evaluate_with_whisper(audio_files_and_targets):
    """Transcribe all WAVs with Whisper and compute WER/CER."""
    import whisper, jiwer
    print("Loading Whisper large-v3-turbo for evaluation …")
    wmodel = whisper.load_model("large-v3-turbo")

    details = []
    for rec in tqdm(audio_files_and_targets, desc="Whisper eval"):
        wav_path = rec["wav_path"]
        target   = rec["target"]
        lang_code = "zh" if rec["lang"] == "chinese" else "ja"
        if not Path(wav_path).exists():
            continue
        result   = wmodel.transcribe(str(wav_path), language=lang_code)
        pred     = result["text"].strip()
        wer_val  = jiwer.wer(target, pred)
        cer_val  = jiwer.cer(target, pred)
        details.append({
            "file":      Path(wav_path).name,
            "lang":      rec["lang"],
            "target":    target,
            "predicted": pred,
            "wer":       wer_val,
            "cer":       cer_val,
        })
    return details


if __name__ == "__main__":
    download_model()
    cosyvoice = load_cosyvoice()

    speakers = sorted([d for d in REF_AUDIO_DIR.iterdir()
                       if d.is_dir() and d.name != "unknown"])
    print(f"Found {len(speakers)} speakers.")

    synth_records = []
    for spk_dir in tqdm(speakers, desc="Speakers"):
        spk_id = spk_dir.name
        ref_wavs = sorted(spk_dir.glob("*.wav"))
        if not ref_wavs:
            continue
        ref_wav = max(ref_wavs, key=lambda p: p.stat().st_size)
        ref_txt_path = ref_wav.with_suffix(".txt")
        if not ref_txt_path.exists():
            continue
        ref_text = ref_txt_path.read_text(encoding="utf-8").strip()

        for lang, sentences in TEST_SENTENCES.items():
            for idx, sentence in enumerate(sentences):
                out_wav = OUTPUT_DIR / f"{spk_id}_{lang}_{idx}.wav"
                if not out_wav.exists():
                    audio, sr = synthesize(
                        cosyvoice, sentence, lang, str(ref_wav), ref_text
                    )
                    if audio is not None:
                        sf.write(str(out_wav), audio, sr)
                synth_records.append({
                    "wav_path": str(out_wav),
                    "lang":     lang,
                    "target":   sentence,
                    "speaker":  spk_id,
                })

    details = evaluate_with_whisper(synth_records)

    avg_wer = sum(d["wer"] for d in details) / len(details) if details else 0
    avg_cer = sum(d["cer"] for d in details) / len(details) if details else 0

    print(f"\n{'='*50}")
    print(f"CosyVoice2-0.5B  WER={avg_wer:.4f}  CER={avg_cer:.4f}  n={len(details)}")
    print(f"{'='*50}")

    out = {"wer": avg_wer, "cer": avg_cer, "n": len(details), "details": details}
    out_path = Path(__file__).parent / "baseline_results/cosyvoice2_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {out_path}")
