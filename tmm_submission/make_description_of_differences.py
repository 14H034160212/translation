"""Generate IEEE TCSVT 'Description of Differences' PDF for the conference extension.

IEEE TCSVT requires authors of journal extensions of prior conference papers to
submit a separate "summary of differences" document detailing what is new in the
journal version relative to the conference paper, per the IEEE policy on prior
publication and the TCSVT Information for Authors.
"""
from fpdf import FPDF


class Doc(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8,
                  "TCSVT submission -- Summary of Differences -- "
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
               "Submitted to IEEE Transactions on Circuits and Systems for "
               "Video Technology (TCSVT)",
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
  "the file labelled \"Conference Extension Paper -- Original "
  "Presentation\". The conference paper is a 4-page IEEE two-column "
  "manuscript; the present journal submission is a substantially "
  "expanded full-length treatment that contains new methodology, new "
  "experiments, an entirely new system stage, and a multi-model "
  "benchmark not present in the conference version.")
p(pdf,
  "Quantitatively, the journal manuscript body contains 6,571 words "
  "vs.\\ 1,830 words in the conference paper (a 3.59x increase), with "
  "8 tables and 1 algorithm vs.\\ 3 tables and 0 algorithms in the "
  "conference paper. We estimate the proportion of net new material at "
  "approximately 71% by word count, well in excess of the new-material "
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
  "a frame-rate ablation, a confidence-adaptive fusion mechanism, "
  "any evaluation against NLLB-200 or Qwen2.5-7B baselines, "
  "chrF++ / COMET / METEOR / TER / BERTScore translation metrics, "
  "any speech synthesis stage, any voice-cloning experiments, or any "
  "speaker-similarity / WER analysis of synthesised audio.")

# ---- 3. New material in the journal version ----
h1(pdf, "3. New Material Introduced in the Journal Version")
p(pdf,
  "The TCSVT submission extends the conference paper along the following "
  "six explicit dimensions. New material is concentrated in Sections "
  "3 (Method), 4 (Experiments) and 5 (Discussion). By word-count "
  "audit of the two LaTeX sources, we estimate net new material at "
  "approximately 71% of the journal manuscript body. Of this, roughly "
  "50% of the journal manuscript consists of categorically new "
  "sections that have no counterpart in the conference paper "
  "(e.g.\\ Speech Synthesis stage, Frame Rate Ablation, Adaptive "
  "Fusion experiments, TTS Comparison, multi-model OCR / translation "
  "benchmarks, Computational Cost analysis, Discussion section), and "
  "the remaining ~21% comes from substantial rewriting and expansion "
  "of carried-over content.")

h2(pdf, "(1) Comprehensive OCR model benchmark (NEW)")
p(pdf,
  "We benchmark seven OCR backbones never compared in the conference "
  "paper -- four large vision-language models (Qwen3-VL, Qwen2-VL, "
  "InternVL2, GOT-OCR2.0) and three traditional engines "
  "(RapidOCR, EasyOCR, TrOCR) -- against Whisper-medium ASR. The "
  "conference paper used Qwen2-VL only.")

h2(pdf, "(2) Frame-rate ablation study (NEW)")
p(pdf,
  "We add a systematic ablation of frame sampling rate from 1 to 5 fps, "
  "demonstrating that 5 fps is essential for high subtitle recall in "
  "fast-paced short dramas. The conference paper used a single fixed "
  "sampling rate without ablation.")

h2(pdf, "(3) Confidence-adaptive multimodal fusion (NEW METHOD)")
p(pdf,
  "We introduce a new fusion strategy that gates between Whisper "
  "transcriptions and Qwen3-VL OCR using Whisper's segment-level "
  "log-probabilities (avg_logprob), replacing the fixed cosine-similarity "
  "threshold of the conference paper. The new strategy improves the "
  "composite recognition score by +10.8% over the ASR-only baseline.")

h2(pdf, "(4) Substantially improved translation model and benchmark "
        "(NEW)")
p(pdf,
  "We replace the Qwen2.5-3B translator with a LoRA fine-tuned Qwen3-VL "
  "model. The best BLEU rises from 9.84 (conference) to 29.99 (journal), "
  "a 3.05x improvement. The best chrF++ rises from 29.99 to 39.34 and "
  "the best COMET from 0.6437 to 0.870. We additionally report METEOR, "
  "TER and BERTScore, none of which appeared in the conference paper. "
  "We add comparisons against NLLB-200 and Qwen2.5-7B baselines that "
  "were not part of the conference evaluation. A further ablation "
  "isolates the contribution of visual context and shows it adds +3.35 "
  "BLEU over text-only translation.")

h2(pdf, "(5) Complete Japanese speech synthesis stage (NEW STAGE)")
p(pdf,
  "An entire third pipeline stage is new relative to the conference "
  "paper. We benchmark zero-shot Japanese voice cloning with three "
  "TTS systems (GPT-SoVITS, F5-TTS, EdgeTTS) over 12 speakers and "
  "report re-transcription Word Error Rate plus a speaker-similarity "
  "ablation. GPT-SoVITS achieves WER 1.17, the lowest of the three. "
  "No speech synthesis component existed in the conference paper.")

h2(pdf, "(6) End-to-end three-stage benchmark on the same corpus (NEW)")
p(pdf,
  "The annotated Chinese-Japanese short-drama corpus is the same as "
  "the conference version (79 paired episodes, 130.56 minutes, 3,692 "
  "subtitle segments); we do not claim a corpus-size increase. The "
  "journal version additionally documents and uses 12 speaker-ID "
  "annotations from the corpus to drive zero-shot voice cloning in the "
  "new TTS stage. Crucially, the journal version is the first "
  "evaluation of all three stages (recognition, translation, "
  "synthesis) on a single corpus end-to-end, yielding -- to the best "
  "of our knowledge -- the first end-to-end multimodal benchmark for "
  "Chinese short-drama localisation. The conference paper benchmarked "
  "only the first two stages.")

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
  "new (multi-model, multi-fps, confidence-adaptive) experimental "
  "protocol. We estimate carried-over content at roughly 29% of the "
  "journal manuscript by word count.")

# ---- 5. Self-citation ----
h1(pdf, "5. Self-Citation")
p(pdf,
  "The conference paper is cited explicitly throughout Section 1 "
  "(\"Relation to prior work\") and at the relevant points in Sections 3, "
  "4 and 5 (bibliography entry \"an2026icassp\"). TCSVT operates a "
  "single-blind review process, so the citation appears in full author "
  "form; no anonymisation of the self-citation is required.")

# ---- 6. Compliance ----
h1(pdf, "6. IEEE Policy Compliance Statement")
p(pdf,
  "Per the IEEE Policy on Prior Publication and the TCSVT Information "
  "for Authors, the present manuscript constitutes a substantial "
  "extension of the ICASSP 2026 paper. The new material -- a 7-model "
  "OCR benchmark, a frame-rate ablation, a confidence-adaptive fusion "
  "algorithm, an expanded translation benchmark with Qwen3-VL backbone "
  "and additional metrics (METEOR, TER, BERTScore), an entirely new "
  "speech-synthesis stage with three TTS systems and 12 speakers, a "
  "computational-cost analysis, and a Discussion section with "
  "limitations and qualitative error analysis -- represents "
  "approximately 71% net new content by word-count audit, materially "
  "above the new-material expectation for a journal extension. The "
  "manuscript is not under simultaneous consideration at any other "
  "venue. The conference version is properly cited and its scope "
  "clearly demarcated in Section 1 of the journal manuscript "
  "(\"Relation to prior work\") and again in the present document.")

out = "/data/home/qbao775/translation/tcsvt_submission/" \
      "description_of_differences.pdf"
pdf.output(out)
print("Wrote", out)
