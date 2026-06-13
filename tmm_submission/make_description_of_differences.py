"""Generate IEEE TMM 'Description of Differences' PDF for the conference extension.

IEEE TMM requires authors of journal extensions of prior conference papers to
submit a separate "summary of differences" document detailing what is new in the
journal version relative to the conference paper, per the IEEE policy on prior
publication and the TMM Information for Authors.

All figures in this document are kept consistent with the current manuscript
(main.tex): eight OCR backbones, eleven translation baselines, BLEU/chrF++/COMET
metrics, four TTS systems plus a blind human listening study, and the
fine-tuning scaling study (4B/8B/12B).
"""
from fpdf import FPDF


class Doc(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8,
                  "TMM submission -- Summary of Differences -- "
                  "Page %d" % self.page_no(),
                  align="C")


def h1(pdf, text):
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(2)
    pdf.multi_cell(0, 7, text)
    pdf.ln(1)


def h2(pdf, text):
    pdf.set_font("Helvetica", "B", 11)
    pdf.ln(1)
    pdf.multi_cell(0, 6, text)


def p(pdf, text):
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.2, text)
    pdf.ln(1)


def bullet(pdf, items):
    pdf.set_font("Helvetica", "", 10)
    for it in items:
        pdf.cell(5)
        pdf.cell(4, 5, "-", ln=0)
        pdf.multi_cell(0, 5.2, it)


pdf = Doc(format="A4")
pdf.set_margins(left=20, top=18, right=20)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# ---- Title ----
pdf.set_font("Helvetica", "B", 15)
pdf.multi_cell(0, 7,
               "Summary of Differences from the Prior Conference Paper",
               align="C")
pdf.ln(2)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 5,
               "Manuscript: An End-to-End Multimodal System for Subtitle "
               "Recognition and Chinese-Japanese Translation with Speech "
               "Synthesis in Short Dramas",
               align="C")
pdf.multi_cell(0, 5,
               "Submitted to IEEE Transactions on Multimedia (TMM)",
               align="C")
pdf.ln(4)

# ---- 1. Prior conference publication ----
h1(pdf, "1. Prior Conference Publication")
p(pdf,
  "An earlier version of this work was accepted to and presented at the "
  "2026 IEEE International Conference on Acoustics, Speech and Signal "
  "Processing (ICASSP 2026), held 4-8 May 2026 in Barcelona, Spain, "
  "under the title \"An End-to-End Multimodal System for Subtitle "
  "Recognition and Chinese-Japanese Translation in Short Dramas\". "
  "The accepted camera-ready version is included in this submission as "
  "the file labelled \"Original Conference Paper\". The conference paper "
  "is a 4-page IEEE two-column manuscript; the present journal submission "
  "is a substantially expanded full-length treatment that adds new "
  "methodology, a multi-model benchmark, an entirely new system stage "
  "(speech synthesis), a human perceptual study, and cross-stage "
  "analyses not present in the conference version.")
p(pdf,
  "Quantitatively, the journal manuscript body contains approximately "
  "10,000 words vs.\\ about 1,830 words in the conference paper, with "
  "15 tables and 1 algorithm vs.\\ 3 tables and 0 algorithms in the "
  "conference paper. We estimate the proportion of net new material at "
  "approximately 80% by word count, well in excess of the new-material "
  "expectation under the IEEE Policy on Prior Publication for journal "
  "extensions of conference papers.")

# ---- 2. Scope of conference paper ----
h1(pdf, "2. Scope of the Conference Paper")
p(pdf,
  "The conference paper introduced a preliminary two-stage pipeline for "
  "Chinese short-drama subtitle localisation, namely:")
bullet(pdf, [
    "A first OCR+ASR fusion module that combined a single VLM "
    "(Qwen2-VL) for visual subtitle recognition with Whisper for "
    "audio transcription, fused via a fixed cosine-similarity threshold.",
    "A LoRA fine-tuned Qwen2.5-3B Chinese-to-Japanese translation model, "
    "evaluated on a small held-out test set with BLEU as the only metric "
    "(reported BLEU = 9.84).",
    "An initial pilot corpus of Chinese short-drama episodes with "
    "Japanese subtitle annotations.",
])
p(pdf,
  "The conference paper did NOT include: a multi-model OCR benchmark, "
  "a frame-rate ablation, a confidence-adaptive (log-probability gated) "
  "fusion mechanism, a cross-validated translation evaluation, any "
  "comparison against the eleven translation baselines listed below, "
  "chrF++ or COMET translation metrics, any speech synthesis stage, any "
  "voice-cloning experiments, any human perceptual study, or any "
  "cross-stage error-propagation analysis.")

# ---- 3. New material in the journal version ----
h1(pdf, "3. New Material Introduced in the Journal Version")
p(pdf,
  "The TMM submission extends the conference paper along the following "
  "explicit dimensions. New material is concentrated in Sections 3 "
  "(Method), 4 (Experiments) and 5 (Discussion). We estimate net new "
  "material at approximately 80% of the journal manuscript body, of "
  "which roughly half consists of categorically new sections that have "
  "no counterpart in the conference paper (Speech Synthesis stage and "
  "human listening study, Frame-Rate Ablation, Adaptive-Fusion grid "
  "search and ablation, multi-model OCR and translation benchmarks, "
  "fine-tuning scaling study, cross-stage error-propagation analysis, "
  "Computational-Cost analysis, and the Discussion section).")

h2(pdf, "(1) Comprehensive OCR model benchmark (NEW)")
p(pdf,
  "We benchmark eight OCR backbones never compared in the conference "
  "paper -- five large vision-language models (Qwen3-VL, Qwen2-VL, "
  "InternVL2, GOT-OCR2.0, Florence-2) and three traditional engines "
  "(RapidOCR, EasyOCR, TrOCR) -- against Whisper-medium ASR. The "
  "conference paper used Qwen2-VL only.")

h2(pdf, "(2) Frame-rate ablation study (NEW)")
p(pdf,
  "We add a systematic ablation of frame sampling rate from 1 to 5 fps, "
  "demonstrating that 5 fps is essential for high subtitle recall in "
  "fast-paced short dramas (recall rises from 27.7% at 1 fps to the "
  "5 fps reference). The conference paper used a single fixed sampling "
  "rate without ablation.")

h2(pdf, "(3) Confidence-adaptive multimodal fusion (NEW METHOD)")
p(pdf,
  "We introduce a two-parameter fusion strategy that gates between "
  "Whisper transcriptions and Qwen3-VL OCR using Whisper's segment-level "
  "log-probabilities (a confidence gate theta_p) together with an OCR "
  "similarity gate theta_s, with both parameters selected by a 10x7 "
  "grid search on a held-out validation subset. This replaces the fixed "
  "single cosine-similarity threshold of the conference paper and "
  "improves the composite recognition score by +10.8% over the ASR-only "
  "baseline. A unified-harness ablation characterises the gating rule "
  "against degenerate and static-threshold baselines.")

h2(pdf, "(4) Cross-validated translation benchmark and fine-tuning "
        "scaling study (NEW)")
p(pdf,
  "We replace the single Qwen2.5-3B translator with a LoRA fine-tuned "
  "Qwen3-VL-4B model evaluated under 5-fold cross-validation "
  "(BLEU 15.70 +/- 4.42, chrF++ 20.78, COMET 0.843; a +32% relative gain "
  "over the same model's zero-shot baseline). We establish a "
  "comprehensive benchmark of eleven baselines absent from the "
  "conference paper -- zero-shot multimodal VLMs (InternVL3-8B, "
  "Qwen3.5-9B, Gemma-4-12B, Qwen3-VL-4B), text-only LLMs (Qwen2.5-7B, "
  "Qwen3-8B, Qwen2.5-3B), and dedicated MT models (M2M-100 1.2B, "
  "SeamlessM4T-v2-large, NLLB-200) -- all evaluated under a standardised "
  "strip_nums + ja-mecab protocol with BLEU, chrF++ and COMET. We "
  "further fine-tune at multiple scales (Qwen3-8B reaching BLEU 19.69 "
  "and Gemma-4-12B reaching BLEU 30.75, the best overall), showing that "
  "domain LoRA fine-tuning helps consistently across the 4B-12B range. "
  "A controlled ablation isolates the contribution of visual context, "
  "showing it adds +3.35 BLEU in the zero-shot setting but yields no "
  "benefit once domain fine-tuning is applied.")

h2(pdf, "(5) Japanese speech synthesis stage with human study (NEW STAGE)")
p(pdf,
  "An entire third pipeline stage is new relative to the conference "
  "paper. We benchmark zero-shot Japanese voice cloning with four TTS "
  "systems (GPT-SoVITS, CosyVoice2-0.5B, F5-TTS, EdgeTTS) over 12 "
  "speakers, reporting Whisper-based intelligibility (character error "
  "rate and MeCab-tokenised word error rate) and Resemblyzer "
  "speaker-similarity, plus a reference-duration ablation. We "
  "additionally conduct a blind human listening study with five "
  "Japanese-proficient raters (naturalness MOS and speaker-similarity "
  "preference, with inter-rater reliability analysis). GPT-SoVITS is the "
  "most intelligible voice-cloning system and matches the commercial "
  "EdgeTTS in human-rated naturalness while dominating perceived speaker "
  "similarity. No speech-synthesis component existed in the conference "
  "paper.")

h2(pdf, "(6) End-to-end analysis on the same corpus (NEW)")
p(pdf,
  "The annotated Chinese-Japanese short-drama corpus is the same as the "
  "conference version (79 paired episodes, 130.56 minutes, 3,692 "
  "subtitle segments); we do not claim a corpus-size increase. The "
  "journal version additionally uses the 12 speaker-ID annotations to "
  "drive zero-shot voice cloning in the new TTS stage, and is the first "
  "evaluation of all three stages (recognition, translation, synthesis) "
  "on a single corpus end-to-end. New analyses include a jieba "
  "part-of-speech study of which errors the fusion corrects (60% of "
  "proper-noun substitutions) and a cross-stage error-propagation study "
  "(both simulated and a direct end-to-end evaluation showing fusion "
  "recovers +2.24 chrF++ over ASR-only), together with an end-to-end "
  "computational-cost / real-time-factor profile. The conference paper "
  "benchmarked only the first two stages with no propagation analysis.")

# ---- 4. Reused material ----
h1(pdf, "4. Material Carried Over from the Conference Paper")
p(pdf,
  "Material carried over from the conference paper is limited to: the "
  "high-level problem framing in the introduction, parts of the related "
  "work coverage of OCR / ASR / NMT (TTS related work is entirely new), "
  "the original two-channel OCR+ASR-then-translate concept, the dataset "
  "itself (79 paired episodes, 130.56 minutes, 3,692 subtitle segments), "
  "the high-level annotation protocol, and Figure 2 (segment-per-episode "
  "distribution). All carried-over text has been rewritten to fit the "
  "longer journal format and to integrate with the new contributions "
  "enumerated above. None of the experimental results from the "
  "conference paper are reproduced verbatim: every recognition and "
  "translation number in the journal manuscript is recomputed under the "
  "new (multi-model, multi-fps, confidence-adaptive, cross-validated) "
  "experimental protocol. We estimate carried-over content at roughly "
  "20% of the journal manuscript by word count.")

# ---- 5. Self-citation ----
h1(pdf, "5. Self-Citation")
p(pdf,
  "The conference paper is cited explicitly in Section 1 (\"Relation to "
  "prior work\") and at the relevant points in Sections 3, 4 and 5 "
  "(bibliography entry \"an2026icassp\"). TMM operates a single-blind "
  "review process, so the citation appears in full author form; no "
  "anonymisation of the self-citation is required.")

# ---- 6. Compliance ----
h1(pdf, "6. IEEE Policy Compliance Statement")
p(pdf,
  "Per the IEEE Policy on Prior Publication and the TMM Information for "
  "Authors, the present manuscript constitutes a substantial extension "
  "of the ICASSP 2026 paper. The new material -- an eight-model OCR "
  "benchmark, a frame-rate ablation, a two-parameter confidence-adaptive "
  "fusion algorithm with grid search and ablation, a cross-validated "
  "eleven-baseline translation benchmark with BLEU/chrF++/COMET and a "
  "fine-tuning scaling study (4B/8B/12B), an entirely new speech-"
  "synthesis stage with four TTS systems and a blind human listening "
  "study, a cross-stage error-propagation analysis, a computational-cost "
  "analysis, and an expanded Discussion with limitations and qualitative "
  "error analysis -- represents approximately 80% net new content by "
  "word-count audit, materially above the new-material expectation for a "
  "journal extension. The manuscript is not under simultaneous "
  "consideration at any other venue. The conference version is properly "
  "cited and its scope clearly demarcated in Section 1 of the journal "
  "manuscript (\"Relation to prior work\") and again in the present "
  "document. The authors used a generative-AI assistant to accelerate "
  "experiment scripting and manuscript editing; all methodological "
  "choices, result interpretation, and verification were performed and "
  "are vouched for by the authors.")

out = "/data/home/qbao775/translation/tmm_submission/" \
      "description_of_differences.pdf"
pdf.output(out)
print("Wrote", out)
