#!/usr/bin/env python3
"""
Re-evaluate TTS intelligibility with a linguistically valid protocol.

Fixes reviewer issue M2 (WER>1 pathology): the original evaluation ran
jiwer.wer() on unsegmented Japanese strings, so each sentence counted as
a single "word" and any mismatch produced WER >= 1.0.

New protocol:
  1. Transcribe with Whisper-medium (language='ja'), matching the ASR stage.
  2. Normalise: NFKC, strip punctuation/whitespace.
  3. CER  = jiwer.cer on normalised character strings (primary metric).
  4. WER  = jiwer.wer on MeCab-tokenised text (sacrebleu ja-mecab),
            space-joined (secondary metric, now well-defined).

Systems:
  GPT-SoVITS  tts_results/speaker{N}_japanese_{S}.wav   (6 speakers)
  F5-TTS      tts_results_f5tts/speaker{N}_japanese_{S}.wav
  CosyVoice2  baseline_results/cosyvoice2/{N}_japanese_{S}.wav
  EdgeTTS     baseline_results/edgetts/{N}_japanese_{S}.mp3

Output: baseline_results/tts_reeval_mecab.json
"""
import json
import re
import unicodedata
from pathlib import Path

import torch
import whisper
from jiwer import cer, wer
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab

ROOT = Path(__file__).parent
SENTENCES = [
    "これは私の履歴書です、どうぞご覧ください。",
    "宋氏グループの面接は私にとって非常に重要です。",
    "母の死の真相を必ず突き止めます。",
]
SYSTEMS = {
    "GPT-SoVITS": (ROOT / "tts_results",                "speaker{n}_japanese_{s}.wav"),
    "F5-TTS":     (ROOT / "tts_results_f5tts",          "speaker{n}_japanese_{s}.wav"),
    "CosyVoice2": (ROOT / "baseline_results/cosyvoice2","{n}_japanese_{s}.wav"),
    "EdgeTTS":    (ROOT / "baseline_results/edgetts",   "{n}_japanese_{s}.mp3"),
}

PUNCT = re.compile(r"[\s。、．，,.!！?？「」『』…・〜~―\-:：;；()（）\"'’”]")
mecab = TokenizerJaMecab()

def norm(text):
    return PUNCT.sub("", unicodedata.normalize("NFKC", text))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)

    results = {}
    for sys_name, (d, pattern) in SYSTEMS.items():
        rows = []
        for n in range(1, 13):
            for s in range(3):
                f = d / pattern.format(n=n, s=s)
                if not f.exists():
                    continue
                hyp = model.transcribe(str(f), language="ja")["text"].strip()
                ref = SENTENCES[s]
                ref_n, hyp_n = norm(ref), norm(hyp)
                c = cer(ref_n, hyp_n) if ref_n else None
                ref_tok = mecab(ref_n)
                hyp_tok = mecab(hyp_n) if hyp_n else ""
                w = wer(ref_tok, hyp_tok)
                rows.append({"speaker": n, "sent": s, "file": f.name,
                             "ref": ref, "hyp": hyp, "cer": c, "wer": w})
                print(f"{sys_name} spk{n} s{s}: CER={c:.3f} WER={w:.3f} | {hyp[:40]}")
        if rows:
            results[sys_name] = {
                "n": len(rows),
                "avg_cer": sum(r["cer"] for r in rows) / len(rows),
                "avg_wer": sum(r["wer"] for r in rows) / len(rows),
                "rows": rows,
            }

    out = ROOT / "baseline_results/tts_reeval_mecab.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY (Whisper-medium, NFKC+punct-strip, mecab WER) ===")
    print(f"{'System':<14} {'CER':>7} {'WER':>7} {'n':>4}")
    for k, v in results.items():
        print(f"{k:<14} {v['avg_cer']:>7.3f} {v['avg_wer']:>7.3f} {v['n']:>4}")
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
