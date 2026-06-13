#!/usr/bin/env python3
"""
Unified-format comparison of static-threshold fusion vs the adaptive
(logprob-gated) fusion — reviewer issue M4.

All configurations are evaluated with the SAME fuse() implementation and
the SAME segment-joined text metrics, so the numbers are directly
comparable (unlike Table 4 in the previous draft, where the Adaptive row
came from the production pipeline with a different text format).

Configurations (on the full eligible corpus):
  Never Replace            θ_s=101  (gate irrelevant)
  Always Replace           θ_p=0.0, θ_s=0
  Static θ_s=40/60/80      θ_p=0.0  (logprob gate always open)
  Adaptive (ours)          θ_p=-0.2, θ_s=80
  Adaptive (alt)           θ_p=-0.2, θ_s=60   — for completeness

Usage:
    ./venv/bin/python run_static_vs_adaptive.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_fusion_grid_search import (
    compute_metrics, fuse, find_valid_episodes, load_episode,
)

CONFIGS = [
    ("Never Replace",            0.0, 101),
    ("Always Replace",           0.0, 0),
    ("Static theta_s=40",        0.0, 40),
    ("Static theta_s=60",        0.0, 60),
    ("Static theta_s=80",        0.0, 80),
    ("Adaptive tp=-0.2 ts=60",  -0.2, 60),
    ("Adaptive tp=-0.2 ts=80",  -0.2, 80),
]

def main():
    episodes = find_valid_episodes()
    print(f"Eligible episodes: {len(episodes)}")
    data = [load_episode(ep) for ep in episodes]

    results = []
    for name, tp, ts in CONFIGS:
        preds, refs = [], []
        for segs, ocr, gt in data:
            preds.append(fuse(segs, ocr, tp, ts))
            refs.append(gt)
        m = compute_metrics(" ".join(preds), " ".join(refs))
        results.append({"config": name, "theta_p": tp, "theta_s": ts, **m})
        print(f"{name:<26} BLEU={m['bleu']:6.2f}  chrF++={m['chrf']:6.2f}  "
              f"CER={m['cer']:.3f}  CharAcc={m['char_acc']:.3f}  "
              f"Comp={m['composite']:.4f}")

    out = Path("speech2text/static_vs_adaptive_unified.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
