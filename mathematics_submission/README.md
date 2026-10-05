# Mathematics (MDPI) submission package

**Target:** *Mathematics* Special Issue "Mathematical Foundations in NLP: Applications and Challenges" (Section: Mathematics and Computer Science).

**Authors:** Jing An ¹,† ; Qiming Bao ²,†,* (corresponding) ; Jinhua Su ³,⁴
(† equal contribution / shared first authorship; * corresponding author, qiming.bao@auckland.ac.nz)

## Files
- `main.tex` — manuscript in **MDPI class format** (porting the TMM content to MDPI).
- `refs.bib` — bibliography (same references; MDPI compiles it with `mdpi.bst`).
- `fig_1_v2.png` — Figure 1 (system architecture).
- `fig_2.png` — used by the Supplementary Materials (per-episode subtitle distribution).
- `supplementary.tex` — Supplementary Materials (Algorithm S1, Tables S1--S9, Figures S1--S2), **standalone** (compiles on its own with XeLaTeX).
- `supplementary.pdf` — local preview (compiled with a fallback CJK font; re-compile on Overleaf with Noto for the final version).
- `cover_letter.md` — cover letter for *Mathematics* (fill in the date).

## How to compile (IMPORTANT)
`main.tex` targets the **official MDPI *Mathematics* LaTeX template**, which provides
`Definitions/mdpi.cls`, `mdpi.bst`, and the journal logos. It will **not** compile
without that template.

1. Open the **MDPI Mathematics** template on Overleaf.
2. Replace its main `.tex` body with `main.tex` here (keep the `Definitions/` folder),
   and add `refs.bib`, `fig_1_v2.png`.
3. **Compile with XeLaTeX** (Menu → Compiler → XeLaTeX). The qualitative examples
   contain a few Chinese/Japanese characters rendered via `xeCJK`; pdfLaTeX will fail.
4. Verify tables/figures render, the bibliography builds, and section numbering is correct.

## TODO before submitting (author to complete / verify)
- [ ] **Confirm the Special Issue is still open** (its stated deadline was 2025-12-31).
- [ ] **APC**: Mathematics charges an Article Processing Charge — confirm funding.
- [ ] **Author consent**: confirm all authors (and the co-authors removed relative to
      the ICASSP/TMM versions) agree to this author list and order.
- [ ] Verify ORCID iDs for all three authors in the MDPI submission form.
- [ ] Fill the submission date in `cover_letter.md`.
- [x] Supplementary Materials prepared (`supplementary.tex`; preview `supplementary.pdf`).
      Re-compile on Overleaf with XeLaTeX + Noto for the final PDF.
- [ ] If the MDPI submission form asks about prior submissions, answer honestly.

## Notes
- MSC 2020 codes used: 68T50 (NLP), 68T07 (artificial neural networks / deep learning),
  68T05 (learning and adaptive systems). Adjust if the editors prefer others.
- Author Contributions (CRediT) in `main.tex` are a reasonable default — edit to match
  the actual division of work among the three authors.
