# Cover Letter — IEEE TMM Submission

**To:** Editor-in-Chief, IEEE Transactions on Multimedia (TMM)

**From:** Jing An (corresponding author), on behalf of all co-authors

Jing An — School of AI and Language Sciences, Beijing International Studies University, Beijing 100024, China — jing.an@bisu.edu.cn

**Authorship:** Qiming Bao is the first author; Jing An is the corresponding author.

**Date:** June 29, 2026

**Manuscript title:** *An End-to-End Multimodal System for Subtitle Recognition and Chinese–Japanese Translation with Speech Synthesis in Short Dramas*

---

Dear Editor,

We are pleased to submit our manuscript, listed above, for consideration as a Regular Paper in *IEEE Transactions on Multimedia*. The paper presents what is, to the best of our knowledge, the first end-to-end multimodal localization system specifically designed for Chinese short-form drama (*duanju*), spanning the tightly coupled stages of visual subtitle recognition (frame-level OCR), audio transcription with adaptive multimodal fusion, neural machine translation, and zero-shot Japanese voice cloning for dubbed audio.

## Why TMM is the appropriate venue

Short-form drama localization is fundamentally a multimedia problem: the subtitle text, the audio waveform, and the visual frame are jointly constrained, and progress on any single modality in isolation is insufficient for a deployable end-to-end pipeline. Our paper contributes (i) a confidence-adaptive OCR/ASR fusion algorithm that gates between the visual and audio channels using ASR log-probabilities; (ii) a domain-adapted vision–language translation component, with a controlled study of how visual context affects translation in zero-shot versus fine-tuned settings; (iii) a controlled comparison of four zero-shot Japanese voice cloning systems combining automatic intelligibility/similarity metrics with a blind human listening study; and (iv) a manually annotated 79-episode cross-modal benchmark with end-to-end error-propagation analysis and real-time-factor profiling. These contributions sit squarely within the TMM scope of multimodal signal processing, cross-modal analysis and fusion, multimedia systems, and applied multimodal machine learning, and the work spans vision, language, and speech in a way that is unlikely to find a natural home at any single-modality venue.

## Relation to prior conference publication

This manuscript is a substantial extension of our preliminary conference paper accepted to **IEEE ICASSP 2026** (Session MMSP-P23), which introduced an early version of the OCR+ASR fusion idea and a small-scale LoRA fine-tuned Qwen2.5-3B translation model on a subset of the data. The present journal version extends that work along six independent dimensions, contributing well beyond the new-material expectation under the IEEE Policy on Prior Publication for journal extensions of conference papers. A separate **Summary of Differences** document detailing the extensions has been uploaded with this submission as required by IEEE policy.

1. **Comprehensive recognition benchmark**: eight OCR backbones evaluated under matched conditions (Qwen3-VL, Qwen2-VL, InternVL2, GOT-OCR2.0, Florence-2, RapidOCR, EasyOCR, TrOCR) versus the single Qwen2-VL configuration in the conference version.
2. **Frame-rate ablation**: a new study quantifying the 1 fps → 5 fps recall gap from 27.7% → 100%, absent from the conference paper.
3. **Upgraded fusion algorithm** (Algorithm 1): replaces the conference paper's fixed similarity threshold with a two-parameter confidence-adaptive gate using Whisper's segment log-probabilities, selected by grid search and formalized as new pseudocode.
4. **Cross-validated translation benchmark and fine-tuning scaling study**: LoRA fine-tuned **Qwen3-VL-4B** achieves 5-fold cross-validated **BLEU 15.70±4.42 / chrF++ 20.78±2.81 / COMET 0.843**, a +25% relative gain over its own per-fold zero-shot mean (and far above the BLEU 9.84 of the conference model). We benchmark **eleven** baselines (incl. M2M-100, SeamlessM4T-v2, NLLB-200, InternVL3-8B, Qwen3.5-9B, Gemma-4-12B, Qwen3-8B) and fine-tune at multiple scales, with Gemma-4-12B reaching BLEU 30.75±1.84, showing domain fine-tuning helps across the 4B–12B range.
5. **Complete speech-synthesis stage**: four zero-shot TTS systems (GPT-SoVITS, CosyVoice2-0.5B, F5-TTS, EdgeTTS) compared with both automatic metrics and a **blind five-rater human listening study**, plus a reference-duration ablation; **entirely new** relative to the conference paper.
6. **New analyses**: visual-context translation ablation (+3.35 BLEU zero-shot; no benefit under fine-tuning), cross-stage error-propagation analysis, computational cost / real-time-factor profiling, and a 100-segment qualitative error taxonomy.

## Authorship and changes since the conference version

The author list of this journal extension differs from the ICASSP 2026 conference paper, reflecting the substantial new contributions described above. We note this change explicitly for transparency:

- **Qiming Bao** (first author; not on the conference paper) led the journal extension — designing and conducting the entirely new speech-synthesis stage together with its blind five-rater human listening study, the cross-validated eleven-baseline translation benchmark and multi-scale (4B/8B/12B) fine-tuning study, the cross-stage error-propagation analysis, and the preparation of this manuscript.
- **Michael Witbrock** (last author; not on the conference paper) provided senior academic supervision and guidance on the methodology, analysis, and presentation of the extended study.
- The remaining co-authors (Jing An, Rui-Yang Ju, Haofei Chang, Jinhua Su, Yanbing Bai, Xin Qu) contributed to the original conference system and to its extension here.
- The author order reflects these relative contributions. All authors have read and approved the manuscript and agreed on the author order and on the corresponding author (Jing An).

## Originality and ethical compliance

- This manuscript is original work; it has not been previously published, and it is not under consideration at any other journal or conference.
- All listed co-authors have read and approved the submission and have agreed on author order.
- The dataset was provided by a commercial entertainment company under a research-use agreement (Section *Ethics Statement*). All performers consented to broadcast distribution; human annotators participated voluntarily and were compensated.
- We acknowledge the dual-use risk of zero-shot voice cloning and have chosen not to release reference-audio-to-cloned-voice mappings; only short, content-neutral synthetic samples will be released for inspection (see Supplementary Materials).
- All authors declare **no competing financial or non-financial interests** related to this work.

## Suggested reviewers (optional)

We do not propose specific reviewers, but we suggest the editorial board consider experts in any of the following overlapping areas: (i) video subtitling and multimedia localization, (ii) multimodal fusion for speech and video understanding, (iii) low-resource and domain-adapted neural machine translation for video content, and (iv) zero-shot voice cloning and TTS evaluation for video dubbing.

## Compliance with TMM submission requirements

- **Format:** IEEE `IEEEtran` class, `journal` option (two-column, single-blind).
- **References:** IEEE Reference Format (`IEEEtran.bst`).
- **Keywords:** IEEE keywords block on the first page.
- **Summary of differences:** uploaded as a separate document for the journal extension of our ICASSP 2026 paper, per IEEE Policy on Prior Publication.
- **Supplementary materials:** README, configuration files, per-episode metric breakdowns, translation example pairs, and short anonymized synthesis samples are provided in the supplementary archive.
- **Reproducibility, Data Availability, and Ethics statements:** provided as separate sections at the end of the manuscript.

We thank you and the reviewers in advance for considering our submission and look forward to your feedback.

Sincerely,

**Jing An** — corresponding author

School of AI and Language Sciences, Beijing International Studies University

1 Dingfuzhuang Nanli, Chaoyang District, Beijing 100024, China

jing.an@bisu.edu.cn

on behalf of all co-authors: Qiming Bao (first author), Jing An (corresponding author), Rui-Yang Ju, Haofei Chang, Jinhua Su, Yanbing Bai, Xin Qu, and Michael Witbrock
