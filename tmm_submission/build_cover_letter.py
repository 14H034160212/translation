#!/usr/bin/env python3
"""Render cover_letter.md into cover_letter.pdf using fpdf2.

Targeted converter for the cover letter we actually wrote: handles the
markdown subset we use (#/##/### headings, **bold**, *italic*, bullet
lists, numbered lists, horizontal rules, plain paragraphs, [text](url)
links). Not a general-purpose md->pdf renderer.
"""

import re
from pathlib import Path

from fpdf import FPDF

SRC = Path(__file__).parent / "cover_letter.md"
DST = Path(__file__).parent / "cover_letter.pdf"

PAGE_W_MM = 210  # A4
LEFT = 25
RIGHT = 25
TOP = 22
BOTTOM = 22
USABLE = PAGE_W_MM - LEFT - RIGHT


class CoverLetter(FPDF):
    def header(self):
        pass

    def footer(self):
        pass


pdf = CoverLetter(format="A4")
pdf.set_margins(left=LEFT, top=TOP, right=RIGHT)
pdf.set_auto_page_break(auto=True, margin=BOTTOM)
pdf.add_page()


# ---- inline rendering --------------------------------------------------

INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
INLINE_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`([^`]+)`")


def tokenize_inline(text: str):
    """Return a list of (style, text) tokens for a single line of text.

    style is one of: "", "B", "I", "BI", "U" (link, rendered underlined).
    """
    # First, remove link markup and replace with sentinel tokens we can split.
    # We use a placeholder so the link text itself can still receive bold/italic.
    placeholders = []

    def link_repl(m):
        placeholders.append(m.group(1))
        return f"\x00LINK{len(placeholders) - 1}\x00"
    s = INLINE_LINK.sub(link_repl, text)

    # Replace inline code with a placeholder we re-emit as plain mono later
    # (cover letter doesn't actually use inline code, but keep handling).
    code_holders = []

    def code_repl(m):
        code_holders.append(m.group(1))
        return f"\x00CODE{len(code_holders) - 1}\x00"
    s = INLINE_CODE.sub(code_repl, s)

    # Now build a token list by scanning for ** and * markers.
    out = []
    pos = 0
    bold = False
    italic = False
    while pos < len(s):
        # check for opening/closing markers
        if s.startswith("**", pos):
            bold = not bold
            pos += 2
            continue
        if s[pos] == "*" and (pos + 1 >= len(s) or s[pos + 1] != "*"):
            italic = not italic
            pos += 1
            continue
        # find next marker
        nxt = len(s)
        for m in ("**", "*"):
            j = s.find(m, pos)
            if j != -1 and j < nxt:
                nxt = j
        chunk = s[pos:nxt]
        if chunk:
            style = ("B" if bold else "") + ("I" if italic else "")
            out.append((style, chunk))
        pos = nxt

    # Re-expand placeholders.
    expanded = []
    for style, chunk in out:
        # Process LINK and CODE placeholders inside chunk.
        cur = chunk
        parts = []
        while True:
            m = re.search(r"\x00(LINK|CODE)(\d+)\x00", cur)
            if not m:
                if cur:
                    parts.append((style, cur))
                break
            head = cur[:m.start()]
            if head:
                parts.append((style, head))
            kind = m.group(1)
            idx = int(m.group(2))
            if kind == "LINK":
                parts.append(((style + "U").replace("UU", "U"),
                              placeholders[idx]))
            else:  # CODE
                parts.append((style, code_holders[idx]))
            cur = cur[m.end():]
        expanded.extend(parts)
    return expanded


def write_inline(tokens, font_size=11, leading=5.0):
    """Write a flowing paragraph from inline tokens at the current y."""
    pdf.set_font_size(font_size)
    line_h = leading
    for style, chunk in tokens:
        # underline = link token
        underline = "U" in style
        font_style = style.replace("U", "")
        pdf.set_font("Helvetica", font_style, font_size)
        if underline:
            # fpdf2 supports the `U` style style flag too.
            pdf.set_font("Helvetica", font_style + "U", font_size)
        # write handles wrapping at word boundaries
        pdf.write(line_h, chunk)
    pdf.ln(line_h)


def heading(text, level):
    sizes = {1: 15, 2: 12.5, 3: 11.5}
    space_before = {1: 4, 2: 6, 3: 5}
    space_after = {1: 3, 2: 2, 3: 2}
    pdf.ln(space_before[level])
    pdf.set_font("Helvetica", "B", sizes[level])
    pdf.multi_cell(USABLE, sizes[level] * 0.55,
                   re.sub(r"\*\*", "", text))
    pdf.ln(space_after[level])


def paragraph(text):
    tokens = tokenize_inline(text)
    pdf.set_font("Helvetica", "", 11)
    write_inline(tokens, font_size=11, leading=5.6)
    pdf.ln(2.5)


def hrule():
    pdf.ln(2)
    y = pdf.get_y()
    pdf.set_draw_color(140, 140, 140)
    pdf.set_line_width(0.2)
    pdf.line(LEFT, y, PAGE_W_MM - RIGHT, y)
    pdf.ln(4)


def bullet_list(items):
    pdf.set_font("Helvetica", "", 11)
    for item in items:
        pdf.set_x(LEFT + 4)
        pdf.cell(4, 5.6, "-", ln=0)
        # render rest of line with inline formatting using write+wrap
        x_text = LEFT + 8
        pdf.set_x(x_text)
        # Save margins, narrow temporarily so write() wraps within text col.
        pdf.set_left_margin(x_text)
        tokens = tokenize_inline(item)
        write_inline(tokens, font_size=11, leading=5.4)
        pdf.set_left_margin(LEFT)
        pdf.ln(0.5)
    pdf.ln(2)


def numbered_list(items):
    pdf.set_font("Helvetica", "", 11)
    for i, item in enumerate(items, 1):
        pdf.set_x(LEFT + 2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(7, 5.6, f"{i}.", ln=0)
        pdf.set_font("Helvetica", "", 11)
        x_text = LEFT + 9
        pdf.set_x(x_text)
        pdf.set_left_margin(x_text)
        tokens = tokenize_inline(item)
        write_inline(tokens, font_size=11, leading=5.4)
        pdf.set_left_margin(LEFT)
        pdf.ln(0.5)
    pdf.ln(2)


# ---- block parser ------------------------------------------------------

def parse_and_render(md_text):
    lines = md_text.splitlines()
    i = 0
    n = len(lines)
    para_buf = []

    def flush_para():
        if para_buf:
            text = " ".join(s.strip() for s in para_buf).strip()
            if text:
                paragraph(text)
            para_buf.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if re.fullmatch(r"-{3,}", stripped):
            flush_para()
            hrule()
            i += 1
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            heading(m.group(2), level)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_para()
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                m_num = re.match(r"^\d+\.\s+(.*)$", lines[i].strip())
                item_text = m_num.group(1)
                j = i + 1
                while j < n and lines[j].startswith("   ") and lines[j].strip():
                    item_text += " " + lines[j].strip()
                    j += 1
                items.append(item_text)
                i = j
            numbered_list(items)
            continue

        if stripped.startswith("- "):
            flush_para()
            items = []
            while i < n and lines[i].strip().startswith("- "):
                item_text = lines[i].strip()[2:]
                j = i + 1
                while j < n and lines[j].startswith("  ") and lines[j].strip():
                    item_text += " " + lines[j].strip()
                    j += 1
                items.append(item_text)
                i = j
            bullet_list(items)
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    flush_para()


_LATIN1_SUBS = {
    "—": "--",  # em dash
    "–": "-",   # en dash
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...",
    " ": " ",   # non-breaking space (latin-1 ok but normalise)
    "→": "->",  # rightwards arrow
}


def _to_latin1(text):
    for k, v in _LATIN1_SUBS.items():
        text = text.replace(k, v)
    # Drop anything still outside latin-1 to avoid encoder crash.
    return text.encode("latin-1", "ignore").decode("latin-1")


def main():
    md = SRC.read_text(encoding="utf-8")
    md = _to_latin1(md)
    parse_and_render(md)
    pdf.output(str(DST))
    print("Wrote", DST)


if __name__ == "__main__":
    main()
