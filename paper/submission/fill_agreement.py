"""Fill the Springer Contributor Agreement template (content controls + address/e-mail cells). Signature and date stay empty.
python fill_agreement.py template.docx out.docx"""
import sys
import docx

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TITLE = "DriveSense: An Explainable Baseline for Early Collision Anticipation from Dashcam Video Using Calibration-Free Cues"
AUTHORS = "Siddhi Dhoke, Nakul Bhadade"
CORR = "Siddhi Dhoke"
ADDRESS = "Indian Institute of Information Technology Nagpur, Nagpur, Maharashtra, India"
EMAIL = "sidii0080@gmail.com"

d = docx.Document(sys.argv[1])
body = d.element.body
rows = list(d.tables[1].rows)
values = {6: TITLE, 10: AUTHORS, 14: CORR}
for ri, val in values.items():
    tr = rows[ri]._tr
    sdts = tr.findall(W + "sdt")
    assert sdts, f"no content control in row {ri}"
    sdt = sdts[0]
    pr = sdt.find(W + "sdtPr")
    ph = pr.find(W + "showingPlcHdr")
    if ph is not None:
        pr.remove(ph)
    ts = list(sdt.iter(W + "t"))
    ts[0].text = val
    for t in ts[1:]:
        t.text = ""
# any remaining unfilled placeholders (should be none that matter)
left = [s for s in body.iter(W + "sdt") if s.find(W + "sdtPr").find(W + "showingPlcHdr") is not None]
print("placeholders left:", [("".join(t.text or "" for t in s.iter(W + "t"))) for s in left])

# address / e-mail cells (table 4): put the value in the last cell of each row
t4 = d.tables[4]
for r, val in ((0, ADDRESS), (1, EMAIL)):
    cell = t4.rows[r].cells[-1]
    p = cell.paragraphs[0]
    for run in p.runs:
        run.text = ""
    if p.runs:
        p.runs[0].text = val
    else:
        p.add_run(val)
d.save(sys.argv[2])
print("saved", sys.argv[2])
