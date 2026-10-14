#!/usr/bin/env python3
"""
Reproduce every listening-study statistic reported in the paper from the
anonymised rating files (raters R1-R5; no names or session identifiers):

  human_eval/listening_naturalness_anonymized.csv     160 naturalness ratings
  human_eval/listening_similarity_votes_anonymized.csv 40 similarity votes

Reports: MOS +- SD per system; Wilcoxon signed-rank tests paired by
rater-item (continuity-corrected, as in the paper); paired t-interval for
GPT-SoVITS - EdgeTTS; rater-clustered bootstrap interval; exact binomial test
on decisive similarity votes; ICC(2,1), ICC(2,k); Krippendorff's ordinal alpha;
per-rater means.

Usage: python3 human_eval/analyze_listening_study.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
SYSTEMS = ["GPT-SoVITS", "EdgeTTS", "F5-TTS", "CosyVoice2"]

nat = pd.read_csv(HERE / "listening_naturalness_anonymized.csv")
votes = pd.read_csv(HERE / "listening_similarity_votes_anonymized.csv")
P = nat.pivot_table(index=["rater", "item"], columns="system", values="naturalness")

print("== Naturalness (MOS +- SD, n per system)")
for s in SYSTEMS:
    print(f"  {s:11s} {P[s].mean():.2f} +- {P[s].std(ddof=1):.2f}  (n={P[s].size})")

print("\n== Wilcoxon signed-rank, paired by rater-item (continuity-corrected)")
for a, b in combinations(SYSTEMS, 2):
    w = stats.wilcoxon(P[a], P[b], correction=True, method="approx")
    print(f"  {a} vs {b}: z={w.zstatistic:.2f}, p={w.pvalue:.3g}")

d = P["GPT-SoVITS"] - P["EdgeTTS"]
half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
print(f"\n== GPT-SoVITS - EdgeTTS: mean {d.mean():.2f}, SD {d.std(ddof=1):.2f}, "
      f"paired 95% t-interval [{d.mean() - half:.2f}, {d.mean() + half:.2f}]")
rng = np.random.default_rng(0)
raters = P.index.get_level_values(0).unique()
boot = [np.concatenate([d.loc[r].values for r in rng.choice(raters, len(raters))]).mean()
        for _ in range(10_000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"   rater-clustered bootstrap 95% interval [{lo:.2f}, {hi:.2f}] (10^4 replicates, seed 0)")

print("\n== Similarity votes")
counts = votes.vote_system.value_counts()
print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
g, f = counts.get("GPT-SoVITS", 0), counts.get("F5-TTS", 0)
print(f"  GPT-SoVITS vs F5-TTS: {g} vs {f}, exact binomial p={stats.binomtest(g, g + f, 0.5).pvalue:.3f}")

# Inter-rater reliability: subjects = (item, system), raters = R1..R5
M = nat.pivot_table(index=["item", "system"], columns="rater", values="naturalness").values
n, k = M.shape
gm = M.mean()
ms_r = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
ms_c = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
ms_e = (((M - gm) ** 2).sum() - ms_r * (n - 1) - ms_c * (k - 1)) / ((n - 1) * (k - 1))
icc21 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
icc2k = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)

vals = sorted(set(M.flatten()))
cnt = {v: (M == v).sum() for v in vals}
def d_ord(c, kk):
    lo_, hi_ = min(c, kk), max(c, kk)
    s = sum(cnt[g_] for g_ in vals if lo_ <= g_ <= hi_) - (cnt[lo_] + cnt[hi_]) / 2
    return s * s
N = M.size
Do = sum(d_ord(M[i, a], M[i, b]) for i in range(n) for a in range(k) for b in range(k) if a != b) / (N * (k - 1))
De = sum(cnt[c] * cnt[kk] * d_ord(c, kk) for c in vals for kk in vals) / (N * (N - 1))
print(f"\n== Reliability: ICC(2,1)={icc21:.3f}, ICC(2,k)={icc2k:.3f}, "
      f"Krippendorff ordinal alpha={1 - Do / De:.3f}")

print("\n== Per-rater mean naturalness")
print(nat.groupby(["rater", "system"]).naturalness.mean().unstack()[SYSTEMS].round(2).to_string())
