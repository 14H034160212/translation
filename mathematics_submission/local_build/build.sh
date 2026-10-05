#!/usr/bin/env bash
#
# Local preview build of ../main.tex against the bundled MDPI class.
#
#   Usage:   ./build.sh
#   Output:  ./main.pdf   (genuine MDPI layout)
#
# Requires a XeTeX engine. We use `tectonic` (https://tectonic-typesetting.github.io).
# If tectonic is not on PATH, set TECTONIC=/path/to/tectonic before running.
#
# NOTE: this is a *preview* build. It uses a community mirror of the MDPI class
# (an older 2020 version, patched so its EPS logos become the bundled PDFs, and
# with a compatibility shim for a few newer macros). The journal / CC-BY logos
# are placeholders and the CJK font may fall back if Noto Serif CJK SC is absent.
# For the final submission PDF, use the current Overleaf MDPI Mathematics
# template with XeLaTeX.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

TECTONIC="${TECTONIC:-tectonic}"
command -v "$TECTONIC" >/dev/null 2>&1 || {
  echo "ERROR: '$TECTONIC' not found. Install tectonic or set TECTONIC=/path/to/tectonic." >&2
  exit 1
}

# bring in the manuscript source + assets
cp ../main.tex ../refs.bib ../fig_1_v2.png .

# compatibility shim for macros the bundled (2020) mdpi.cls lacks but the
# current MDPI template provides (injected into this local copy only)
python3 - <<'PY'
s = open('main.tex').read()
shim = (r"""\providecommand{\TitleCitation}[1]{}
\providecommand{\AuthorCitation}[1]{}
\providecommand{\orcidA}{}
\providecommand{\orcidB}{}
\providecommand{\institutionalreview}[1]{\par\noindent\textbf{Institutional Review Board Statement:} #1\par}
\providecommand{\informedconsent}[1]{\par\noindent\textbf{Informed Consent Statement:} #1\par}
\providecommand{\dataavailability}[1]{\par\noindent\textbf{Data Availability Statement:} #1\par}
""")
s = s.replace(r'\usepackage{xeCJK}', shim + r'\usepackage{xeCJK}', 1)
open('main.tex', 'w').write(s)
PY

# fall back to a CJK-capable font if Noto Serif CJK SC is not installed
if ! fc-list 2>/dev/null | grep -qi "Noto Serif CJK SC"; then
  if [ -f /usr/share/fonts/google-droid/DroidSansFallback.ttf ]; then
    sed -i 's|\\setCJKmainfont{Noto Serif CJK SC}|\\setCJKmainfont{DroidSansFallback.ttf}[Path=/usr/share/fonts/google-droid/]|' main.tex
    echo "note: Noto Serif CJK SC not found -> using DroidSansFallback for CJK."
  fi
fi

"$TECTONIC" main.tex
echo "Built: $HERE/main.pdf"
