"""Generate IEEE TMM Title Page PDF (separate file requested by editorial office)."""
from fpdf import FPDF


class TitlePage(FPDF):
    def header(self):
        pass

    def footer(self):
        pass


pdf = TitlePage(format="A4")
pdf.set_margins(left=20, top=20, right=20)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# --- Title ---
pdf.set_font("Helvetica", "B", 15)
pdf.multi_cell(0, 8,
               "An End-to-End Multimodal System for "
               "Subtitle Recognition\n"
               "and Chinese-Japanese Translation with "
               "Speech Synthesis in Short Dramas",
               align="C")
pdf.ln(8)

# --- Authors ---
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6,
               "Qiming Bao (2), Jing An (1, *), "
               "Rui-Yang Ju (3),\n"
               "Haofei Chang (4), Jinhua Su (4, 5), "
               "Yanbing Bai (4), Xin Qu (1), "
               "Michael Witbrock (6)",
               align="C")
pdf.ln(6)

# --- Affiliations ---
pdf.set_font("Helvetica", "I", 10)
affils = [
    "(1) School of AI and Language Sciences, "
    "Beijing International Studies University, Beijing 100024, China",
    "(2) The University of Auckland, Auckland 1010, New Zealand",
    "(3) Graduate School of Informatics, "
    "Kyoto University, Kyoto 606-8501, Japan",
    "(4) Center for Applied Statistics, School of Statistics, "
    "Renmin University of China, Beijing 100872, China",
    "(5) Simashuhui Ltd., Beijing 100086, China",
    "(6) School of Computer Science, "
    "The University of Auckland, Auckland 1010, New Zealand",
]
for line in affils:
    pdf.multi_cell(0, 5, line, align="C")
pdf.ln(8)

# --- Manuscript info ---
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 6, "Manuscript Information", ln=1)
pdf.set_font("Helvetica", "", 10)
pdf.cell(40, 5, "Manuscript ID:", ln=0)
pdf.cell(0, 5, "[to be assigned by ScholarOne]", ln=1)
pdf.cell(40, 5, "Journal:", ln=0)
pdf.cell(0, 5,
         "IEEE Transactions on Multimedia (TMM)",
         ln=1)
pdf.ln(4)

# --- Author notes ---
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 6, "Author Notes", ln=1)
pdf.set_font("Helvetica", "", 10)
notes = [
    "  asterisk (*): Corresponding author for manuscript correspondence "
    "(Jing An).",
]
for n in notes:
    pdf.multi_cell(0, 5, n)
pdf.ln(4)

# --- ORCIDs ---
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 6, "ORCID iDs", ln=1)
pdf.set_font("Helvetica", "", 10)
orcids = [
    ("Qiming Bao", "0000-0002-1000-7383"),
    ("Rui-Yang Ju", "0000-0003-2240-1377"),
    ("Haofei Chang", "0009-0003-6690-9065"),
    ("Jinhua Su", "0000-0002-6344-0607"),
    ("Yanbing Bai", "0000-0001-5223-9425"),
]
for name, oid in orcids:
    pdf.cell(45, 5, "  " + name, ln=0)
    pdf.cell(0, 5, oid, ln=1)
pdf.ln(4)

# --- Corresponding author ---
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 6, "Corresponding Author (manuscript correspondence)", ln=1)
pdf.set_font("Helvetica", "", 10)
ca = [
    ("Name:", "Jing An"),
    ("Affiliation:",
     "School of AI and Language Sciences, "
     "Beijing International Studies University"),
    ("Address:",
     "1 Dingfuzhuang Nanli, Chaoyang District, Beijing 100024, China"),
    ("Email:", "jing.an@bisu.edu.cn"),
]
for k, v in ca:
    pdf.cell(28, 5, "  " + k, ln=0)
    pdf.multi_cell(0, 5, v)
pdf.ln(4)

out = "/data/home/qbao775/translation/tmm_submission/title_page.pdf"
pdf.output(out)
print("Wrote", out)
