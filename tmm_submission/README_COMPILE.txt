COMPILATION NOTE -- PLEASE READ
================================

This manuscript MUST be compiled with the XeLaTeX engine (not pdfLaTeX).

Reason: the discussion section contains a small number of Chinese and
Japanese characters (qualitative translation/recognition examples), which
are rendered through the xeCJK package. The xeCJK package requires XeLaTeX;
compiling with pdfLaTeX will fail on these characters.

Files in this archive (the complete main manuscript):
  - main.tex          : the manuscript source (documentclass: IEEEtran, journal)
  - refs.bib          : bibliography (compiled with IEEEtran.bst)
  - fig_1_v2.png      : Figure 1 (system architecture)

Build sequence:
  xelatex main && bibtex main && xelatex main && xelatex main

Notes:
  - The first line of main.tex carries the directive "% !TEX program = xelatex".
  - The IEEEtran class and IEEEtran.bst are standard and available on the
    submission system; they are not bundled here.
  - The manuscript is 10 pages including references (author biographies are
    intentionally disabled for the initial submission and do not count toward
    the page limit).
