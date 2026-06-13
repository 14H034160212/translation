#!/usr/bin/env python3
"""Aggregate 5-fold CV results for the two revision pilots and print a summary.

  - Gemma-4-12B FT       : baseline_results/gemma4_12b_ft_fold{0-4}.json
  - Qwen3-VL multimodal FT: baseline_results/qwen3vl_multimodal_ft_fold{0-4}.json

Reference baselines (already in paper):
  - Gemma-4-12B ZS per-fold : 25.69 24.96 27.57 26.38 26.59  (mean 26.24)
  - Qwen3-VL-4B text FT     : 19.68 19.75  9.58 16.53 12.94  (mean 15.70)
"""
import json
from pathlib import Path
import statistics as st

RES = Path("baseline_results")

GEMMA_ZS   = [25.69, 24.96, 27.57, 26.38, 26.59]
QWENVL_TXT = [19.68, 19.75, 9.58, 16.53, 12.94]

def collect(prefix):
    bleus, chrfs = [], []
    for f in range(5):
        p = RES / f"{prefix}_fold{f}.json"
        if not p.exists():
            bleus.append(None); chrfs.append(None); continue
        d = json.loads(p.read_text())
        bleus.append(d["BLEU"]); chrfs.append(d["chrF++"])
    return bleus, chrfs

def fmt(xs):
    return " ".join(f"{x:.2f}" if x is not None else "  -- " for x in xs)

def summary(name, bleus, chrfs, ref_bleu):
    done = [b for b in bleus if b is not None]
    print(f"\n=== {name} ===")
    print(f"  per-fold BLEU : {fmt(bleus)}")
    print(f"  per-fold chrF+: {fmt(chrfs)}")
    if len(done) == 5:
        mb, sb = st.mean(bleus), st.stdev(bleus)
        mc, sc = st.mean(chrfs), st.stdev(chrfs)
        print(f"  >> BLEU  = {mb:.2f} +/- {sb:.2f}")
        print(f"  >> chrF++ = {mc:.2f} +/- {sc:.2f}")
        print(f"  reference per-fold mean = {st.mean(ref_bleu):.2f}")
        deltas = [b - r for b, r in zip(bleus, ref_bleu)]
        print(f"  delta vs ref per-fold   = {fmt(deltas)}  (mean {st.mean(deltas):+.2f})")
    else:
        print(f"  ({len(done)}/5 folds done)")

gb, gc = collect("gemma4_12b_ft")
qb, qc = collect("qwen3vl_multimodal_ft")
summary("Gemma-4-12B FT (5-fold) vs Gemma ZS", gb, gc, GEMMA_ZS)
summary("Qwen3-VL-4B multimodal FT (5-fold) vs text FT", qb, qc, QWENVL_TXT)
