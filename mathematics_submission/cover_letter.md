# Cover Letter — *Mathematics* (MDPI) Submission

**To:** The Editors and Guest Editors, *Mathematics* — Special Issue *"Mathematical Foundations in NLP: Applications and Challenges"* (Section: Mathematics and Computer Science)

**From:** Qiming Bao (corresponding author), on behalf of all co-authors

Qiming Bao — School of Computer Science, University of Auckland, Auckland 1010, New Zealand — qiming.bao@auckland.ac.nz

**Authorship:** Jing An and Qiming Bao contributed equally and share first authorship; Qiming Bao is the corresponding author.

**Date:** [INSERT SUBMISSION DATE]

**Manuscript title:** *An End-to-End Multimodal System for Subtitle Recognition and Chinese–Japanese Translation with Speech Synthesis in Short Dramas*

---

Dear Editors,

We are pleased to submit our manuscript for consideration in the *Mathematics* Special Issue *"Mathematical Foundations in NLP: Applications and Challenges."* The paper presents, to the best of our knowledge, the first end-to-end multimodal localization system for Chinese short-form drama (*duanju*), spanning visual subtitle recognition, confidence-adaptive OCR/ASR fusion, neural machine translation, and zero-shot Japanese voice cloning.

## Fit with the Special Issue

The work sits squarely within *mathematical foundations of NLP and their applications*: (i) we formalize a two-parameter, probabilistically gated OCR/ASR fusion rule driven by segment-level log-probabilities, and select its operating point by a systematic grid search with a stated objective; (ii) we study domain adaptation of neural translation models under a rigorous 5-fold cross-validation protocol with paired significance testing (paired *t*-tests, Wilcoxon signed-rank, bootstrap confidence intervals); and (iii) we quantify cross-stage error propagation as an approximately linear recognition-error-to-translation-quality relation. These contributions combine learning-theoretic modeling with careful empirical and statistical analysis of AI-driven NLP, matching the Special Issue's emphasis on improvements in machine-learning models and novel applications of deep learning in NLP.

## Relation to a prior conference paper

This manuscript is a substantial extension of our conference paper, *An End-to-End Multimodal System for Subtitle Recognition and Chinese–Japanese Translation in Short Dramas*, in Proc. IEEE ICASSP, 2026, which introduced an early OCR+ASR fusion idea and a small LoRA fine-tuned translation model. The prior work is cited in the manuscript. The present version adds: an eight-backbone OCR benchmark and frame-rate ablation; a two-parameter confidence-gated fusion selected by grid search; a cross-validated eleven-configuration translation study with multi-scale (4B/8B/12B) fine-tuning; an entirely new speech-synthesis stage with four TTS systems and a blind human listening study; and a cross-stage error-propagation analysis.

## Originality and ethical compliance

- This manuscript is original work; it has not been previously published, and it is **not currently under consideration at any other journal or conference**.
- All listed authors have read and approved the submission and agreed on the author order and corresponding author.
- Human listening-study participants were adult volunteers who gave informed consent and were compensated; no personally identifying information was collected (see the *Informed Consent* and *Institutional Review Board* statements in the manuscript).
- The dataset was provided by a commercial entertainment company under a research-use agreement (see *Data Availability*).
- The authors declare **no conflicts of interest**.

We thank you and the reviewers for considering our submission.

Sincerely,

**Qiming Bao** — corresponding author

School of Computer Science, University of Auckland, Auckland 1010, New Zealand

qiming.bao@auckland.ac.nz

on behalf of all co-authors: Jing An, Qiming Bao, and Jinhua Su
