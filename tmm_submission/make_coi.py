"""Generate the Conflict of Interest statement PDF for IEEE TMM submission."""
from fpdf import FPDF

TITLE = ("An End-to-End Multimodal System for Subtitle Recognition and "
         "Chinese-Japanese Translation with Speech Synthesis in Short Dramas")
AUTHORS = ("Qiming Bao, Jing An, Rui-Yang Ju, Haofei Chang, Jinhua Su, "
           "Yanbing Bai, Xin Qu, and Michael Witbrock")

pdf = FPDF(format="A4")
pdf.set_margins(left=20, top=20, right=20)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

pdf.set_font("Helvetica", "B", 14)
pdf.multi_cell(0, 8, "Conflict of Interest Statement", align="C")
pdf.ln(4)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "Journal:", ln=1)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 5, "IEEE Transactions on Multimedia (TMM)")
pdf.ln(2)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "Manuscript title:", ln=1)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 5, TITLE)
pdf.ln(2)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "Authors:", ln=1)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 5, AUTHORS)
pdf.ln(6)

pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6,
    "On behalf of all co-authors, I confirm that the authors have no conflict "
    "of interest to disclose. No author has a competing financial or "
    "non-financial interest related to this work.\n\n"
    "This statement is also disclosed within the manuscript (Acknowledgment "
    "section).")
pdf.ln(10)

pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 5, "Corresponding author: Jing An (jing.an@bisu.edu.cn)", ln=1)
pdf.cell(0, 5, "School of AI and Language Sciences, "
               "Beijing International Studies University", ln=1)
pdf.ln(4)
pdf.cell(0, 5, "Date: ____________________", ln=1)

out = "/data/home/qbao775/translation/tmm_submission/conflict_of_interest.pdf"
pdf.output(out)
print("Wrote", out)
