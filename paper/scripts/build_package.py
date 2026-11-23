"""Build IRIS2027_DriveSense_submission.zip (only files that main.tex actually uses), then prove it compiles from a clean copy.
Run from the paper folder after a successful build:  python scripts/build_package.py <path to pdflatex>"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

pdflatex = sys.argv[1] if len(sys.argv) > 1 else "pdflatex"
src = Path("main.tex").read_text(encoding="utf-8")
inputs = re.findall(r"\\input\{([^}]+)\}", src)
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
files = ["main.tex", "svproc.cls"] + inputs + figs
out = Path("submission/IRIS2027_DriveSense_submission.zip")
tmp = Path(tempfile.mkdtemp())
for f in files:
    (tmp / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, tmp / f)
for _ in range(2):
    r = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=tmp, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
pages = re.search(r"\((\d+) pages", r.stdout)
print("clean build OK,", pages.group(1) if pages else "?", "pages")
shutil.copy(tmp / "main.pdf", "main.pdf")
shutil.copy(tmp / "main.pdf", "submission/main.pdf")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files + ["main.pdf"]:
        z.write(tmp / f if f != "main.pdf" else "main.pdf", f)
print("wrote", out, "with", len(files) + 1, "files:", files)
