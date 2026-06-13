# Cover Letter — IEEE TCSVT Submission

**To:** Editor-in-Chief, IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

**From:** Qiming Bao (joint first author and first corresponding author) and Yanbing Bai (second corresponding author), on behalf of all co-authors

Qiming Bao — The University of Auckland, Auckland 1010, New Zealand — qiming.bao@auckland.ac.nz

Yanbing Bai — Center for Applied Statistics, School of Statistics, Renmin University of China, Beijing 100872, China — ybbai@ruc.edu.cn

**Authorship:** Jing An and Qiming Bao are joint first authors (equal contribution); Qiming Bao and Yanbing Bai are joint corresponding authors.

**Date:** [INSERT SUBMISSION DATE]

**Manuscript title:** *An End-to-End Multimodal System for Subtitle Recognition and Chinese–Japanese Translation with Speech Synthesis in Short Dramas*

---

Dear Editor,

We are pleased to submit our manuscript, listed above, for consideration as a Regular Paper in *IEEE Transactions on Circuits and Systems for Video Technology*. The paper presents what is, to the best of our knowledge, the first end-to-end multimodal localization system specifically designed for Chinese short-form drama (*duanju*), spanning the three tightly coupled video-processing stages of visual subtitle recognition (frame-level OCR), audio transcription with adaptive multimodal fusion, neural machine translation, and zero-shot Japanese voice cloning for dubbed audio.

## Why TCSVT is the appropriate venue

Short-form drama localization is fundamentally a circuits-and-systems-for-video problem: the subtitle text, the audio waveform, and the visual frame are jointly constrained, and progress on any single modality in isolation is insufficient for a deployable video pipeline. Our paper contributes (i) a confidence-adaptive OCR/ASR fusion algorithm that gates between modalities using ASR log-probabilities; (ii) a domain-adapted vision–language translation model that exploits visual context for register-aware Chinese–Japanese rendering; (iii) a controlled comparison of three zero-shot Japanese voice cloning systems with a reference-duration ablation; and (iv) a manually annotated 79-episode benchmark with end-to-end real-time-factor profiling. These contributions sit squarely within the TCSVT scope of video signal analysis, video-understanding systems, multimodal video processing, and applied learning for video content, and the work is unlikely to find a natural home at a single-modality venue (vision, NLP, or speech).

## Relation to prior conference publication

This manuscript is a substantial extension of our preliminary conference paper accepted to **IEEE ICASSP 2026** (Session MMSP-P23), which introduced an early version of the OCR+ASR fusion idea and a small-scale LoRA fine-tuned Qwen2.5-3B translation model on a subset of the data. The present journal version extends that work along six independent dimensions, contributing well beyond the new-material expectation under the IEEE Policy on Prior Publication for journal extensions of conference papers. A separate **Summary of Differences** document detailing the extensions has been uploaded with this submission as required by IEEE policy.

1. **Comprehensive recognition benchmark** (Section 5.3): seven OCR backbones evaluated under matched conditions (Qwen3-VL, Qwen2-VL, InternVL2, GOT-OCR2.0, RapidOCR, EasyOCR, TrOCR) versus the single Qwen2-VL configuration in the conference version.
2. **Frame-rate ablation** (Section 5.4): a new study quantifying the 1 fps → 5 fps recall gap from 27.7% → 100%, absent from the conference paper.
3. **Upgraded fusion algorithm** (Section 4.2.3, Algorithm 1): replaces the conference paper's fixed similarity threshold with a confidence-adaptive gate using Whisper's segment log-probabilities, formalized as new pseudocode.
4. **Substantially stronger translation results**: LoRA fine-tuned **Qwen3-VL-4B** achieves **BLEU 29.99 / chrF++ 39.34 / COMET 0.870**, versus the BLEU 9.84 reported in the conference paper. We additionally benchmark NLLB-200 and Qwen2.5-7B baselines.
5. **Complete speech-synthesis stage** (Sections 4.4 and 5.7): GPT-SoVITS, F5-TTS, and EdgeTTS comparison with reference-duration ablation, **entirely new** relative to the conference paper.
6. **New analyses**: visual-context translation ablation (+3.35 BLEU), computational cost / real-time-factor profiling, and a 100-segment qualitative error taxonomy.

## Originality and ethical compliance

- This manuscript is original work; it has not been previously published, and it is not under consideration at any other journal or conference.
- All listed co-authors have read and approved the submission and have agreed on author order.
- The dataset was provided by a commercial entertainment company under a research-use agreement (Section *Ethics Statement*). All performers consented to broadcast distribution; human annotators participated voluntarily and were compensated.
- We acknowledge the dual-use risk of zero-shot voice cloning and have chosen not to release reference-audio-to-cloned-voice mappings; only short, content-neutral synthetic samples will be released for inspection (see Supplementary Materials).
- All authors declare **no competing financial or non-financial interests** related to this work.

## Suggested reviewers (optional)

We do not propose specific reviewers, but we suggest the editorial board consider experts in any of the following overlapping areas: (i) video subtitling and multimedia localization, (ii) multimodal fusion for speech and video understanding, (iii) low-resource and domain-adapted neural machine translation for video content, and (iv) zero-shot voice cloning and TTS evaluation for video dubbing.

## Reviewer-excluded list

We respectfully request that the following individuals not be invited as reviewers due to recent collaboration: *[INSERT NAMES IF ANY; LEAVE EMPTY OTHERWISE]*.

## Compliance with TCSVT submission requirements

- **Format:** IEEE `IEEEtran` class, `journal` option (two-column, single-blind).
- **References:** IEEE Reference Format (`IEEEtran.bst`).
- **Keywords:** IEEE keywords block on the first page.
- **Summary of differences:** uploaded as a separate document for the journal extension of our ICASSP 2026 paper, per IEEE Policy on Prior Publication.
- **Supplementary materials:** README, configuration files, per-episode metric breakdowns, translation example pairs, and short anonymized synthesis samples are provided in the supplementary archive.
- **Reproducibility, Data Availability, and Ethics statements:** provided as separate sections at the end of the manuscript.

We thank you and the reviewers in advance for considering our submission and look forward to your feedback.

Sincerely,

**Qiming Bao** — first corresponding author

The University of Auckland, Auckland 1010, New Zealand

qiming.bao@auckland.ac.nz

**Yanbing Bai** — second corresponding author

Center for Applied Statistics, School of Statistics, Renmin University of China

59 Zhongguancun Street, Haidian District, Beijing 100872, China

ybbai@ruc.edu.cn

on behalf of all co-authors: Jing An (joint first author), Rui-Yang Ju, Haofei Chang, Jinhua Su, and Xin Qu
