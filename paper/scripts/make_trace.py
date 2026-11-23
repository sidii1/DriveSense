"""Writes NUMBERS_TRACE.md: every result macro used in main.tex with its value and the generated file (and so the script
and results file) it comes from, plus the literal numbers that are typed in the text and where they come from.
Run from the paper folder: python scripts/make_trace.py"""
import glob
import re

bs = chr(92)
src = open("main.tex", encoding="utf-8").read()
gens = {f: open(f, encoding="utf-8").read() for f in sorted(glob.glob("generated_*.tex"))}
origin = {
    "generated_pilot.tex": "paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv}",
    "generated_pilot_table.tex": "paper/scripts/pilot.py",
    "generated_eval.tex": "experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json",
    "generated_within.tex": "experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json",
    "generated_extra.tex": "experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv}",
    "generated_runtime.tex": "experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json",
    "generated_tab_main.tex": "04_anticipation_eval.py (Table 2)", "generated_tab_leak.tex": "04_anticipation_eval.py (Table 1)",
    "generated_tab_within.tex": "06_within_clip.py (Table 3)", "generated_tab_control.tex": "04_anticipation_eval.py (not in the paper; kept in repo)",
}
defs = {}
for f, t in gens.items():
    for m in re.finditer(r"\\(?:def\\|newcommand\{\\)([A-Za-z]+)\}?\{(.*?)\}\s*$", t, flags=re.M):
        defs[m.group(1)] = (m.group(2), f)
body = src[src.index(bs + "begin{abstract}"):src.index(bs + "begin{thebibliography}")]
sections = re.split(r"(?=\\section\{|\\subsection\{|\\begin\{abstract\})", body)
out = ["# NUMBERS_TRACE", "",
       "Every result macro used in the abstract and Sections 1-6 of `main.tex`, with its value and origin. "
       "Macros are written by the scripts named below; nothing is typed by hand except the items under *Literal numbers*.", ""]
for sec in sections:
    title = re.match(r"\\(?:sub)?section\{(.*?)\}|\\begin\{abstract\}", sec)
    name = (title.group(1) if title and title.group(1) else "Abstract") if title else None
    if name is None:
        continue
    used = []
    for m in re.finditer(r"\\([A-Z][A-Za-z]+)\b", sec):
        k = m.group(1)
        if k in defs and k not in used:
            used.append(k)
    if not used:
        continue
    out += [f"## {name}", "", "| macro | value | origin |", "|---|---|---|"]
    for k in used:
        v, f = defs[k]
        out.append(f"| `\\{k}` | {v.replace('|', '/')} | {origin.get(f, f)} |")
    out.append("")
out += ["## Literal numbers typed in the text", "",
        "| number | meaning | source |", "|---|---|---|",
        "| 1.19 million | road deaths per year | WHO Global Status Report on Road Safety 2023 (ref. 1) |",
        "| 1,500 / 3,000 / 50 frames / 10 fps / 5 s | CCD composition | CCD README (`data/CarCrash/README_CCD.md`), verified by `00b_build_crash_table.py` and `results/eval/clips_used.csv` |",
        "| 0.25, 960, 0.22W, 0.6 s, 0.02 s^-1, 4 samples | detector / TTC settings | `experiments/exp3_ccd/config.py` |",
        "| Farnebaeck parameters, central region | flow settings | `experiments/exp3_ccd/perception/flow.py` |",
        "| 300 trees, depth 4, lr 0.05 | XGBoost settings | `04_anticipation_eval.py::fit_xgb` |",
        "| 10 % FAR, 1,000 bootstrap, 2 s horizon, 20 % held-out | protocol | `04_anticipation_eval.py` defaults (`--far --boot --horizon`), `06_within_clip.py` constants |",
        "| 14 cues | Holm family size of the audit | `results/temporal_tests.csv` (rows) |",
        "| 3 % | largest relative shift of photometric cues in the audit (sharpness 593 -> 575) | `generated_pilot.tex` (TSharpEarly/TSharpLate) |",
        "| 51,641; 11.2 %; 0.612 [0.595, 0.631]; 0.684 [0.668, 0.701]; +0.072 [0.058, 0.088]; 0.763 [0.749, 0.775]; ~70 % | CRSS 2024 injury model | `experiments/exp1_crss/results/evaluation_metrics.json` (`drivesense_tiers`, `delta_tier2_all_vs_tier1`), `crss_data_summary.json`; 70 % = (0.684-0.5)/(0.763-0.5) |",
        ""]
open("NUMBERS_TRACE.md", "w", encoding="utf-8", newline="\n").write("\n".join(out))
print("NUMBERS_TRACE.md written:", len(out), "lines")
