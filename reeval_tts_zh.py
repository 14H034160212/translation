#!/usr/bin/env python3
"""
Intra-lingual (Chinese reference -> Chinese target) reference condition for the
TTS benchmark, scored with the same protocol as the cross-lingual Japanese
results (reeval_tts_wer_cer.py, compute_speaker_similarity.py):

  1. Transcribe with Whisper-medium (language='zh').
  2. Normalise: NFKC, traditional->simplified (OpenCC t2s), strip punctuation/whitespace.
  3. CER = jiwer.cer on characters.
  4. Speaker similarity = Resemblyzer cosine against the same per-speaker
     reference embedding used for the Japanese outputs.

The Chinese outputs were synthesised by the same scripts and with the same
reference clip per speaker as the Japanese outputs.

Output: baseline_results/tts_reeval_zh.json
"""
import json, re, unicodedata
from pathlib import Path

import jiwer
import numpy as np
import opencc
import torch
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav

from compute_speaker_similarity import get_ref_embedding

ROOT = Path(__file__).parent

SENTENCES = [
    "这是我的个人简历，请您过目。",
    "宋氏集团的面试对我来说非常重要。",
    "我一定会查清楚母亲去世的真相。",
]

SYSTEMS = {
    "GPT-SoVITS": (ROOT / "tts_results",                 "speaker{n}_chinese_{s}.wav", [1, 2, 3, 4, 7, 8]),
    "F5-TTS":     (ROOT / "tts_results_f5tts",           "speaker{n}_chinese_{s}.wav", list(range(1, 13))),
    "CosyVoice2": (ROOT / "baseline_results/cosyvoice2", "{n}_chinese_{s}.wav",        list(range(1, 13))),
}

PUNCT = re.compile(r"[\s。、．，,.!！?？「」『』“”…・〜~―\-:：;；()（）\"'’]")
t2s = opencc.OpenCC("t2s")


def norm(text):
    return PUNCT.sub("", t2s.convert(unicodedata.normalize("NFKC", text)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = whisper.load_model("medium", device=device)
    enc = VoiceEncoder()

    results = {}
    for name, (d, pattern, speakers) in SYSTEMS.items():
        rows = []
        for n in speakers:
            ref_emb = get_ref_embedding(enc, n)
            for s, target in enumerate(SENTENCES):
                f = d / pattern.format(n=n, s=s)
                if not f.exists():
                    continue
                hyp_raw = asr.transcribe(str(f), language="zh")["text"].strip()
                ref, hyp = norm(target), norm(hyp_raw)
                cer = jiwer.cer(ref, hyp) if hyp else 1.0
                emb = enc.embed_utterance(preprocess_wav(str(f)))
                sim = float(np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb)))
                rows.append({"speaker": n, "sent": s, "file": f.name, "ref": target,
                             "hyp": hyp_raw, "cer": cer, "spk_sim": sim})
        cers = [r["cer"] for r in rows]
        sims = [r["spk_sim"] for r in rows]
        results[name] = {"n": len(rows), "avg_cer": float(np.mean(cers)),
                         "median_cer": float(np.median(cers)),
                         "avg_spk_sim": float(np.mean(sims)), "rows": rows}
        print(f"{name:11s} n={len(rows):2d}  CER mean={np.mean(cers):.3f} "
              f"median={np.median(cers):.3f}  Spk-Sim={np.mean(sims):.3f}")

    out = ROOT / "baseline_results/tts_reeval_zh.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
