# Local MDPI preview build

This folder lets you compile `../main.tex` into a **genuine MDPI-layout PDF
locally**, without the Overleaf template, for quick previews.

## What's here
- `Definitions/` — a community mirror of the MDPI class, patched for local use:
  - `mdpi.cls` — the MDPI article class (its EPS logo paths changed to the bundled PDFs).
  - `mdpi.bst`, `journalnames.tex` — bibliography style and journal-name table.
  - `logo-mdpi.pdf`, `logo-orcid.pdf` — real converted logos.
  - `logo-mathematics.pdf`, `logo-ccby.pdf` — **placeholders** (not the real journal / CC-BY logos).
- `build.sh` — copies `../main.tex` + assets here, applies a small compatibility
  shim, and compiles with `tectonic`.

## How to run
```bash
cd mathematics_submission/local_build
./build.sh              # or:  TECTONIC=/path/to/tectonic ./build.sh
# output: main.pdf
```
Requires a XeTeX engine (`tectonic` recommended). On this server:
```bash
TECTONIC=/data/home/qbao775/translation/texenv/bin/tectonic ./build.sh
```

## Important caveats (this is a PREVIEW)
- The bundled `mdpi.cls` is an **older (2020)** mirror; the current Overleaf
  template is newer and renders `\TitleCitation`, `\institutionalreview`,
  `\informedconsent`, `\dataavailability` natively (here they are shimmed).
- The **journal logo and CC-BY logo are placeholders**.
- CJK uses a **fallback font** if `Noto Serif CJK SC` is not installed.

**For the final submission PDF, use the official Overleaf "MDPI Mathematics"
template with XeLaTeX** — that gives the correct class version, logos, and fonts.
This local build is only for fast iteration/preview.
