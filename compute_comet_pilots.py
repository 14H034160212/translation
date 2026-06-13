#!/usr/bin/env python3
"""
Compute per-fold COMET for the two revision pilots, then average — matching the
5-fold CV protocol used for the other FT rows in the paper.

Run with the COMET-capable env:
  CUDA_VISIBLE_DEVICES=5 /data/home/qbao775/translation/venv/bin/python compute_comet_pilots.py

Reads baseline_results/{gemma4_12b_ft,qwen3vl_multimodal_ft}_fold{0-4}.json
Writes baseline_results/{...}_comet_summary.json
"""
import json
import re
import statistics as st
from pathlib import Path

RES = Path("baseline_results")

def strip_nums(t):
    return re.sub(r"\d+:\s*", "", t).strip()

def load_fold(path):
    d = json.loads(Path(path).read_text())
    rows = [r for r in d["results"] if r.get("prediction", "").strip()]
    return ([strip_nums(r["source"])     for r in rows],
            [strip_nums(r["reference"])  for r in rows],
            [strip_nums(r["prediction"]) for r in rows])

def main():
    from comet import download_model, load_from_checkpoint
    mp = download_model("Unbabel/wmt22-comet-da")
    cm = load_from_checkpoint(mp)

    for prefix in ["gemma4_12b_ft", "qwen3vl_multimodal_ft"]:
        per_fold = []
        for f in range(5):
            p = RES / f"{prefix}_fold{f}.json"
            if not p.exists():
                print(f"  [skip] {p.name} missing")
                continue
            srcs, refs, preds = load_fold(p)
            data = [{"src": s, "ref": r, "mt": m} for s, r, m in zip(srcs, refs, preds)]
            out = cm.predict(data, batch_size=8, gpus=1)
            score = out.system_score if hasattr(out, "system_score") else st.mean(out.scores)
            per_fold.append(round(float(score), 4))
            print(f"  {prefix} fold{f}: COMET={score:.4f}  (n={len(data)})")
        if len(per_fold) == 5:
            summ = {"per_fold_comet": per_fold,
                    "COMET": round(st.mean(per_fold), 4),
                    "COMET_std": round(st.stdev(per_fold), 4)}
            (RES / f"{prefix}_comet_summary.json").write_text(json.dumps(summ, indent=2))
            print(f"  >> {prefix}: COMET = {summ['COMET']} +/- {summ['COMET_std']}\n")

if __name__ == "__main__":
    main()
