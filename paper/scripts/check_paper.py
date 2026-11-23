"""Pre-submission checks. Run from the paper folder:  python scripts/check_paper.py
 1. every \\Macro used in main.tex is defined (main.tex or generated_*.tex)
 2. no TODO / TBD left
 3. references are cited in order of first citation and every item is cited
"""
import glob
import re
import sys

bs = chr(92)
src = open("main.tex", encoding="utf-8").read()
gen = "".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob("generated_*.tex")))
body = src[src.index(bs + "begin{document}"):src.index(bs + "begin{thebibliography}")]
head = src[: src.index(bs + "begin{document}")]
ok = True

defined = set(re.findall(r"\\(?:def|newcommand\{?|renewcommand\{?|providecommand\{?)\\?([A-Za-z]+)", src + gen))
defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", src + gen))
used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", body)) | set(re.findall(r"\\([A-Z][A-Za-z]+)\b", gen))
builtin = {"TBD", "TODO", "Large", "LARGE", "IfFileExists", "EvalAvailable", "EvalAvailabletrue", "EvalAvailablefalse"}
standard = {"S", "AA", "ORCID", "Delta", "Rightarrow", "None"}
tex_builtin = {"Delta", "Rightarrow", "Leftrightarrow", "Gamma", "Sigma", "Omega", "Phi", "Pi", "Theta", "Lambda"}
missing = sorted(m for m in used if m not in defined and m not in builtin and m not in standard
                 and m not in tex_builtin)
if missing:
    ok = False
    print("UNDEFINED macros:", missing)

hits = [l for l in (src + gen).splitlines() if re.search(r"TODO|TBD", l) and not l.lstrip().startswith("%")
        and "newcommand{" + bs + "TBD}" not in l and "newcommand{" + bs + "TODO}" not in l]
if hits:
    ok = False
    print("TODO/TBD lines:\n  " + "\n  ".join(hits[:10]))

order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in order:
            order.append(k)
items = re.findall(r"\\bibitem\{([^}]*)\}", src)
if order != items:
    ok = False
    print("reference order mismatch:\n cited:", order, "\n listed:", items)
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
