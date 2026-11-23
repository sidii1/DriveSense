# Prompt for Claude Code — finish the DriveSense IRIS 2027 paper

> Paste everything below the line into Claude Code, opened at the root of the `DriveSense` repo.
> First unzip `DriveSense_IRIS27_paper_package.zip` into the repo as `paper/`, and make sure the CCD videos and annotation files are on disk.

---

You are finishing a Springer conference paper for **IRIS 2027 (Shaastra, IIT Madras; Springer LNEE proceedings)**. **The hard deadline is Wed 25 Nov 2026, 23:59 IST.** Work autonomously, but follow the rules below. They matter more than speed.

## 0. Context (read first)
- Repo: `automotive-vision-pipeline/` (perception code, `experiments/exp1_crss`, `experiments/exp3_ccd`).
- Paper: `paper/main.tex`. It uses the Springer `svproc` class; do **not** edit `svproc.cls`. It must stay at **exactly 10 pages** and compile with `pdflatex main && pdflatex main`. References are inline in `thebibliography`, use Springer "Basic" style, are numbered in order of first citation, carry DOIs and contain no hyperlinks.
- Also read `paper/README_SUBMISSION.md`. It explains how the paper is wired:
  - Every result is a macro or table body loaded from `generated_*.tex`.
  - `scripts/pilot.py` → `generated_pilot*.tex` + `figures/fig_pilot_curves.pdf`.
  - `scripts/04_anticipation_eval.py` (also at `experiments/exp3_ccd/04_anticipation_eval.py`) → `generated_eval.tex`, `generated_tab_main.tex`, `generated_tab_control.tex` and `figures/fig_shap.pdf`.
  - Until 04 runs, values print as a boxed `[TBD]`, and three red `\TODO{}` interpretation notes remain in Section 5.
- Authors (final; do not change): **Siddhi Dhoke** (first, corresponding, sidii0080@gmail.com) and **Nakul Bhadade** (nakulccs@gmail.com), Indian Institute of Information Technology Nagpur, Nagpur, Maharashtra, India.
- Title (fixed, from the accepted abstract): *DriveSense: An Explainable Baseline for Early Collision Anticipation from Dashcam Video Using Calibration-Free Cues*.

## 1. Non-negotiable integrity rules
1. **Never invent, round toward, or hand-type a result.** Every number in the paper must come from a script you ran and a results file it wrote. Keep it that way: extend the generator scripts and the macros. Do not paste numbers into prose.
2. **Report what the data says, even if it is negative or unflattering.** Examples: the leakage AUC is high, fusion does not beat the rule, or the flow effect is not crash-specific. In those cases rewrite the claims to fit. An honest negative result with clean methodology is publishable. A claim the code does not support gets the paper rejected or retracted.
3. **The old synthetic results are banned from the paper.** Anything from `generate_multimodal_dataset()`, the README's "+17% ROC-AUC lift", Model A/B/C, or "NHTSA CRSS fusion" on video must not appear. The paper may only mention them as the labelled "pitfall" paragraph that is already there.
4. **Splits are by clip, never by frame.** Thresholds are fixed on held-out training clips only. The test set is touched once per configuration. Seeds are fixed. Bootstrap CIs are taken over clips.
5. **No third-party images in the paper.** That means no dashcam frames from CCD/KITTI/YouTube and no dashboard screenshots. Our own plots and diagrams only.
6. If something genuinely cannot be done, say so in the final report. Do not quietly drop it or fake it.

## 2. Phase A — environment and data check (≈15 min)
1. Create or activate a venv and `pip install -r experiments/exp3_ccd/requirements_exp3.txt xgboost scikit-learn scipy matplotlib pandas`. Report the GPU or CPU and the frames/s of `01_extract_features.py` on 20 frames.
2. Locate the CCD data: the `Crash-1500` videos, the `Normal` videos, `Crash-1500.txt` and any official train/test list files. Print counts.
3. **Optical flow:** the committed `01_extract_features.py` does **not** produce the `flow_*` columns, yet `results_v1_backup/exp3_v2_test/frame_features_all.csv` contains `flow_div_center`, `flow_div_center_p90`, `flow_div_full` and `flow_mag_center`.
   - Find the local code that produced them. If it is missing, implement `perception/flow.py` with OpenCV Farnebäck and add it to the extractor. Reproduce the four columns on 3 of the pilot clips and check that they match the stored values (correlation > 0.95).
   - Commit the flow code.
   - Fix the `\TODO{confirm the flow method…}` in Sect. 3.3. Describe the exact method, its parameters and the central-region definition. Keep reference `farneback2003` only if Farnebäck is really what was used.

## 3. Phase B — core evaluation (required)
1. Build the crash table (`00b_build_crash_table.py`). Then extract features for **all 1,500 crash and 3,000 normal clips**, resumable, with the same extractor and the same settings for both classes. Write normal clips to a separate output dir, e.g. `EXP3_OUT=results_normal`.
   - If time does not allow the full set, process **all official test clips plus a class-stratified random subset of the training clips**, choosing the largest set that finishes by Tue 24 Nov 18:00 IST. Record exactly which clips were used.
2. Re-run `02_build_video_features.py` and `03_temporal_analysis.py` on the full crash set.
   - Then regenerate the pilot section with `python paper/scripts/pilot.py <crash results dir> paper`, so that Sect. 5.1, Table 1 and Fig. 2 describe the **full** crash set instead of 57 clips.
   - Update the wording ("pilot" → "audit", n values). Keep the 57-clip numbers out of the paper.
3. Run `04_anticipation_eval.py` with `--split-dir` pointing at the official CCD lists. If the script falls back to the seeded split, investigate the list format and fix the parser. Do not accept the fallback silently.
4. Sanity checks, and fix any failures before going further:
   - no clip appears in both train and test;
   - the test FAR is close to the target;
   - shuffling the crash/normal labels gives AP ≈ prevalence and AUC ≈ 0.5;
   - the rule baseline's score is exactly smoothed `max_inv_ttc_path`.

## 4. Phase C — upgrades that strengthen the paper (do them in this order and stop when time runs out)
These raise the paper from "a baseline" to "a finding". Each one must be implemented as code that writes to the results files and from there into `generated_*.tex`.

**C1. Source-leakage deep dive (highest value).** The CCD accident and normal clips may come from different sources.
- (a) Compare low-level container and encoding properties between the two classes: resolution, fps, bitrate/file size per second, codec, letterboxing (black-border rows), JPEG/blockiness or noise level, and mean frame-to-frame difference in the first second.
- (b) Train the first-second leakage classifier on these "nuisance" features alone, and also on scene, kinematic and motion cues.
- (c) Look up the published AP/mTTA of deep anticipation models on CCD from their original papers. Use web search and cite only numbers you verified at the source, with the exact table.
- If nuisance features alone separate the classes well (e.g. AUC > 0.8), state clearly and carefully that part of CCD's crash-vs-normal separability is source-driven and does not depend on the accident. Then propose and report a **source-controlled protocol**: for example, normalize resolution, crop letterbox borders, and match normals on weather/time-of-day where the metadata allows, then re-evaluate.
- If leakage is low, say so: that validates CCD as a benchmark, which is also a contribution.
- Add a compact table with the leakage AUCs per feature set and a CI for each.

**C2. Crash-specific dynamics.** Using the control comparison already in 04:
- (a) Plot event-aligned mean curves of `flow_mag_center`, `has_det` and `max_inv_ttc_path` for crash vs. normal clips, aligned to onset and pseudo-onset with bootstrap bands. Make it black-and-white legible (line styles, not colours) and replace Fig. 2 with it.
- (b) Report effect sizes.

**C3. Diagnose the detector-coverage collapse.** On a stratified sample of ~150 crash clips, log the **raw YOLO detections** (`model.predict`) next to the ByteTrack outputs (`model.track`). Measure how much of the late-window coverage drop comes from the detector and how much from the tracker dropping unconfirmed tracks, and add one sentence plus the numbers. If the tracker is responsible, also evaluate a variant that uses raw detections plus IoU matching for box-growth TTC, and report both.

**C4. A detector-free time-to-contact from flow divergence.** For a looming surface, TTC ≈ 2 / divergence of the image flow in the central region. Add `tau_div = 2/max(div,eps)` (clipped) as a kinematic-free cue in its own group named `motion-TTC`. Evaluate it as a rule baseline and inside the fused model. This directly strengthens contribution (i), calibration-free TTC, because it also survives detector failure. Document the formula in Sect. 3.2 with one numbered equation.

**C5. Operating curves and efficiency.**
- (a) Recall and mTTA as a function of FAR (5–30%) for the rule, kinematic-only and fused models: one small B&W figure.
- (b) Per-frame runtime of each pipeline stage on the machine used, to back the "lightweight" claim.

**C6. Robustness slices.** Split test recall and AP at the fixed FAR by day/night, weather (normal/rain/snow) and ego-involved vs. not, with CIs. Report them in a table only if it fits in the page budget. Otherwise give two sentences and put the full slice table in a supplementary CSV mentioned in the Data and Code Availability paragraph.

Do **not** add new deep models, new datasets or anything that cannot finish and be verified by Tue 24 Nov night.

## 5. Phase D — write-up
1. Replace each of the three red `\TODO{}` notes in Section 5 with 1–3 sentences interpreting the generated numbers. Each sentence must be supported by a table or figure. Update the abstract's last result sentences and the contributions list to match what was actually found, including C1–C4 if they were done.
   - Keep the abstract ≤ 250 words, with no footnotes and no citations.
2. Keep the paper at **exactly 10 pages**. If it overflows, cut prose before figures: merge or condense CRSS Table 4 into text, and shorten Related Work. Never shrink fonts or margins, and never alter the class.
3. Figures: vector PDF, lettering ≥ 6 pt, legible in black and white. Captions go below figures and above tables, with no full stop when the caption is not a sentence. Every figure and table is referenced in the text.
4. References: every cited item is in the list and every listed item is cited. Order by first citation, using the check script below. Add DOIs that you verify and no hyperlinks. Any new reference (e.g. for C1(c) or C4) must be checked at the publisher or DOI page.
5. Update the **Limitations** paragraph to reflect the final scope.
6. Commit everything (scripts, configs, small results CSV/JSON, figures) to the repo on a branch `iris27-final`, and write clear commit messages.
7. Clean the public repo, because the paper points to it:
   - Rewrite `README.md` and `docs/model_comparison_report.md` so the synthetic benchmark is clearly labelled as a synthetic development artifact, the "+17%/+24.5% lift" and the "NHTSA CRSS fusion" claims are removed, and the hard-coded `Conclusion: YES` in the report writer is deleted.
   - Add a "Reproducing the IRIS 2027 paper" section listing the exact commands.

## 6. Phase E — package and verify
1. Run `grep -n "TODO\|TBD" paper/main.tex paper/generated_*.tex`. It must print nothing. Also open the PDF and check that there is no red text and no boxed TBD.
2. `pdfinfo paper/main.pdf` must report exactly 10 pages. Compile from a clean copy of the folder to prove it builds.
3. Reference-order check:
```python
import re; s=open('paper/main.tex').read()
b=s[s.index('\\begin{document}'):s.index('\\begin{thebibliography}')]
o=[];[o.append(k.strip()) for m in re.finditer(r'\\cite\{([^}]*)\}',b) for k in m.group(1).split(',') if k.strip() not in o]
assert o==re.findall(r'\\bibitem\{([^}]*)\}',s), (o,)
```
4. Trace every number: produce `paper/NUMBERS_TRACE.md`. List each numeric claim in the abstract and Sections 5–6 with the file and field it comes from.
5. Build `IRIS2027_DriveSense_submission.zip` containing `main.tex`, `svproc.cls`, all `generated_*.tex`, `figures/` and `main.pdf`.
6. Fill the Springer **Contributor Agreement** `.docx` using python-docx. Leave the signature and date empty. Enter exactly:
   - chapter title as above;
   - authors "Siddhi Dhoke, Nakul Bhadade";
   - corresponding author Siddhi Dhoke, Indian Institute of Information Technology Nagpur, Nagpur, Maharashtra, India, sidii0080@gmail.com.
7. Draft the submission e-mail to `conference@shaastra.org` with the subject `IRIS 2027 – Final Paper Submission – Siddhi Dhoke`. List the attachments, both authors' e-mails, the running head ("DriveSense: Explainable Collision Anticipation from Dashcam Video") and the statement "No third-party material; no permissions required".

## 7. Final report to me
End with a short report:
- what ran, and on how many clips;
- the headline numbers with CIs;
- which of C1–C6 were completed;
- any result that contradicts the accepted abstract, with how the paper now handles it;
- anything still needing a human: signatures, e-mail sending, ORCIDs, acknowledgments.
