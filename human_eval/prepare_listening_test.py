#!/usr/bin/env python3
"""
Prepare audio files for TTS human listening study.

Selects 8 representative speaker-sentence pairs across 4 TTS systems,
copies them into a structured directory, and generates a randomised
stimulus manifest for each of 5 raters.

Run:
    python3 human_eval/prepare_listening_test.py

Output:
    human_eval/stimuli/          — audio files (WAV)
    human_eval/manifest_rater{1-5}.json  — per-rater randomised order
    human_eval/stimulus_legend.txt       — mapping code→system (for scoring)
"""

import json, shutil, random
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = Path(__file__).parent / "stimuli"
OUT.mkdir(exist_ok=True)

# ── 3 sentences used in evaluation ─────────────────────────────────────────
SENTENCES = {
    0: "これは私の履歴書です、どうぞご覧ください。",
    1: "宋氏グループの面接は私にとって非常に重要です。",
    2: "母の死の真相を必ず突き止めます。",
}

# ── 8 test items: (speaker_id, sentence_id)
# GPT-SoVITS covers speakers 1-4, 7, 8 only.
# item 3 changed spk=6→4 (sent=2); item 8 changed spk=12→7 (sent=0) to ensure
# all 4 systems are present for every item.
TEST_ITEMS = [
    (2,  0),   # item 1 — sentence 0
    (4,  1),   # item 2 — sentence 1
    (4,  2),   # item 3 — sentence 2  (spk=4, different sentence from item 2)
    (3,  0),   # item 4 — sentence 0
    (7,  1),   # item 5 — sentence 1
    (8,  2),   # item 6 — sentence 2
    (1,  1),   # item 7 — sentence 1
    (7,  0),   # item 8 — sentence 0  (spk=7, different sentence from item 5)
]

# ── TTS system audio locations ──────────────────────────────────────────────
SYSTEMS = {
    "GPT-SoVITS":    ROOT / "tts_results",           # speakerN_japanese_S.wav (speakers 1-4,7,8 only)
    "F5-TTS":        ROOT / "tts_results_f5tts",      # speakerN_japanese_S.wav
    "CosyVoice2":    ROOT / "baseline_results/cosyvoice2",  # N_japanese_S.wav
    "EdgeTTS":       ROOT / "baseline_results/edgetts",     # N.mp3 (fixed voice, no cloning)
}

# Reference audio dir: data/tts_ref_audio/{speaker_id}/
REF_DIR = ROOT / "data/tts_ref_audio"

def find_synth(system, system_dir, spk, sent):
    """Return path to synthesised wav (or None if unavailable)."""
    if system in ("GPT-SoVITS", "F5-TTS"):
        p = system_dir / f"speaker{spk}_japanese_{sent}.wav"
    elif system == "CosyVoice2":
        p = system_dir / f"{spk}_japanese_{sent}.wav"
    elif system == "EdgeTTS":
        # EdgeTTS uses a fixed voice — no per-speaker clone; pick the mp3
        p = system_dir / f"{spk}.mp3"
    return p if p.exists() else None

def find_reference(spk):
    """Return the first reference wav for this speaker."""
    ref_spk_dir = REF_DIR / str(spk)
    refs = sorted(ref_spk_dir.glob("*.wav"))
    return refs[0] if refs else None

# ── Copy files and build stimulus list ─────────────────────────────────────
stimuli = []  # list of dicts: {item, spk, sent, text, system, file, is_reference}

for item_idx, (spk, sent) in enumerate(TEST_ITEMS, start=1):
    text = SENTENCES[sent]

    # Copy reference audio
    ref_path = find_reference(spk)
    if ref_path:
        ref_out = OUT / f"item{item_idx:02d}_spk{spk}_REF.wav"
        shutil.copy2(ref_path, ref_out)
        stimuli.append({
            "item": item_idx, "speaker": spk, "sentence": sent,
            "text": text, "system": "REFERENCE", "role": "reference",
            "filename": ref_out.name,
        })

    # Copy synthesised audio for each system
    for sys_name, sys_dir in SYSTEMS.items():
        src = find_synth(sys_name, sys_dir, spk, sent)
        if src is None:
            print(f"  MISSING: {sys_name} spk={spk} sent={sent}")
            continue
        ext = src.suffix
        dst_name = f"item{item_idx:02d}_spk{spk}_sent{sent}_{sys_name.replace('-','').replace(' ','')}{ext}"
        shutil.copy2(src, OUT / dst_name)
        stimuli.append({
            "item": item_idx, "speaker": spk, "sentence": sent,
            "text": text, "system": sys_name, "role": "test",
            "filename": dst_name,
        })

print(f"\nCopied {len(stimuli)} audio files to {OUT}")

# ── Build randomised per-rater manifests ────────────────────────────────────
# Each rater sees all 8 items. Within each item, 4 system clips are randomised.
# Raters are assigned different random seeds for order randomisation.
SEEDS = [42, 137, 256, 512, 1024]

for rater_id, seed in enumerate(SEEDS, start=1):
    rng = random.Random(seed)
    manifest = []

    for item_idx, (spk, sent) in enumerate(TEST_ITEMS, start=1):
        text = SENTENCES[sent]

        # Get test clips for this item
        item_clips = [s for s in stimuli if s["item"] == item_idx and s["role"] == "test"]
        rng.shuffle(item_clips)

        # Assign blind codes A/B/C/D
        coded = []
        for code, clip in zip(["A", "B", "C", "D"], item_clips):
            coded.append({
                "code":     code,
                "filename": clip["filename"],
                "system":   clip["system"],   # kept here but NOT in rater-facing doc
            })

        # Reference
        ref = next((s for s in stimuli if s["item"] == item_idx and s["role"] == "reference"), None)

        manifest.append({
            "item":      item_idx,
            "speaker":   spk,
            "sentence":  sent,
            "text":      text,
            "reference": ref["filename"] if ref else None,
            "clips":     coded,
        })

    # Shuffle item order too
    rng.shuffle(manifest)

    manifest_path = Path(__file__).parent / f"manifest_rater{rater_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Rater {rater_id} manifest → {manifest_path.name}")

# ── Write legend (kept secret from raters) ─────────────────────────────────
legend_lines = ["# Stimulus Legend (DO NOT SHARE WITH RATERS)\n"]
for s in stimuli:
    if s["role"] == "test":
        legend_lines.append(f"{s['filename']}  →  {s['system']}")
legend_path = Path(__file__).parent / "stimulus_legend.txt"
legend_path.write_text("\n".join(legend_lines))
print(f"Legend → {legend_path.name}")

print("\nDone. Upload the stimuli/ directory to a shared server or Google Drive,")
print("then paste the links into the Google Form template.")
