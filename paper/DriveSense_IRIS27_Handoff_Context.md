# DriveSense / IRIS 27 — Full Context Handoff

**Purpose of this file:** give a new assistant (with no prior context) everything needed to help Siddhi finish and submit her final research paper for **IRIS 2027 (IIT Madras / Shaastra)**. Snapshot date: **Saturday 21 November 2026**. **Hard deadline: Wednesday 25 November 2026, 11:59 PM** (≈4 days left). Everything below was compiled from one long working session plus the project's saved notes.

---

## 0. Instructions for the assistant reading this

1. **The goal is a finished, submission-ready 10-page paper package by 25 Nov.** Prioritise ruthlessly; every suggestion should be doable in hours, not days.
2. **Communication style:** Siddhi is informal, practical and iterative. She prefers clear, actionable guidance over long prose. Use short sections, tables and checklists. Ask at most one question at a time.
3. **Honesty is non-negotiable.** An earlier version of the project's headline result turned out to be an artifact of synthetic data (see §5). Never let it back into the paper. Only claim what a reproducible script and a results file support.
4. **Do not invent numbers.** Anything marked `[TBD]` below must come from her actual runs. If she pastes results, analyse them; if not, ask.
5. **First thing to do in the new chat:** ask her for the items in §14 ("Unknowns to confirm") — especially the accepted abstract text and the author list.
6. She cannot be assumed to remember every detail of the pipeline code; ask her to paste scripts or outputs when needed. Uploading files works better than pasting for CSV/JSON.

---

## 1. Who Siddhi is

- B.Tech CSE (AI & ML) student at **IIIT Nagpur** (Maharashtra, India).
- Prior tech lead on a real startup (The Consistent Academy, 200+ active users); strong backend/full-stack, "vibe-coding" background being converted into structured knowledge.
- Career target: automotive/tech internships and placements (Bosch, KPIT, Valeo, Tata Elxsi, Tata Technologies), aiming to secure opportunities by 3rd year. This paper is a CV and interview asset, so **rigour and a defensible story matter more than flashy claims**.
- Related earlier work: *Road Risk Predictor* (deployed web app, XGBoost on NHTSA CRSS 2022 records, AUC 0.72) and the lane-detection + YOLOv8 ADAS project on KITTI footage.

---

## 2. The target: IRIS 27 final paper

**Event:** Shaastra International Conference IRIS 27 (International Research Conference on Computer Vision), IIT Madras Research Park, Chennai, **January 2–3, 2027**. Oral or poster presentation in person. ₹50,000 prize pool for best presentations. Selected papers go to **Springer Nature (Lecture Notes in Electrical Engineering)** *after further Springer review* — publication is not guaranteed by selection.

**Status:** her abstract was **selected**. Final paper package is now due.

**Eligibility notes:** up to 7 authors; work must be original, unpublished and not under consideration elsewhere; plagiarism leads to rejection; Springer may run plagiarism software.

### Submission package (email to `conference@shaastra.org`)
Subject line: `IRIS 2027 – Final Paper Submission – [Corresponding Author Name]`

| # | Item | Notes |
|---|---|---|
| 1 | Final **source files** | Word/RTF, or LaTeX with all `.tex`, style files, fonts, images **and the `.bbl`** (`.bib` alone is not enough). Must compile cleanly. |
| 2 | Final **PDF** | Created and checked by her; must match source exactly. |
| 3 | **Contributor Agreement** (Springer copyright form) | Corresponding author prints, fills in chapter title + all author names + corresponding author details, **signs**, scans. Signer must be the corresponding author named in the paper header. She has the `.docx` form. |
| 4 | **Permissions Checklist** + signed permission/release forms | Only if the paper contains third-party content (see §2.3). Templates she has: `Research-Permission-Template-EN.docx`, `Research-Release-Template-EN.docx`. |
| 5 | **Emails** | Corresponding author's email mandatory; all authors' preferred. |
| 6 | Optional | Short running head if title is long; supplementary material; data/code availability statement. |

### 2.1 Formatting rules (from the official guidelines)
- Use Springer's **full-paper proceedings template for electrical engineering / physics / mathematics** — **not** the LNCS (computer science) package, not an abstract template. Download from Springer's "Information for Authors of Springer Proceedings" page. Do not alter styles, margins, spacing or fonts (Times in Word, CMR in LaTeX).
- **Exactly 10 pages.** American English. No page numbers or running heads (Springer adds them).
- Title centred 14 pt bold; 1st-level headings 12 pt bold; 2nd-level 10 pt bold; only two heading levels numbered; Title Case for headings.
- Authors: full names, given name before surname, no abbreviations; affiliations with town/city and country; ORCID encouraged; **exactly one corresponding author**, email mandatory. Names, order, affiliations and corresponding author must be **identical** in the paper, the Contributor Agreement and the submission email; they cannot change after delivery.
- Abstract per template, no footnotes in the abstract.
- Figures: vector where possible; line drawings ≥ 800 dpi; lettering ≥ 6 pt; captions below figures, above tables, 9 pt; must be legible in black-and-white (print is B&W).
- Equations: centred, numbered consecutively `(1), (2)…`, editable (Word Math/MathType or LaTeX environments), punctuated as text.
- Acknowledgments: run-in heading, not a numbered section. Appendices before references.
- **References:** *Basic* style, *Content Only* numbering; cite `[9]`, `[4–6, 9]` in order of first citation; every reference cited and vice versa; DOIs where available, **no hyperlinks**; "et al." beyond six authors.
- Program code/commands in monospaced font.

### 2.2 Originality
Must not be published or submitted elsewhere. Siddhi **has not** submitted to ICAIDIET'26 (dropped), so there is no duplicate-submission problem. She has an older draft `DriveSense_AI_ICAIDIET26_Paper_Draft.docx` that reports the invalid synthetic "lift" — **do not reuse its claims**.

### 2.3 Third-party material (permissions)
- Anything not created by her needs written permission unless public-domain, covered by a suitable licence (check CC NC/ND carefully), or a legal exception.
- **Strong recommendation: use only her own figures/tables** (plots, diagrams, SHAP charts, tables) so no permission forms are needed.
- **Do not put raw frames from CCD, KITTI, Nexar or the YouTube-style demo video** in the paper unless the licence allows publication. As far as known, KITTI is CC BY-NC-SA (non-commercial), which Springer flags as possibly incompatible. Dashcam frames also show other people's faces and number plates (release-form territory). Cite datasets in the references instead.
- A dashboard screenshot is her own software, but if it embeds third-party video frames it triggers the same issue. Keep the dashboard **out of the paper** (see §9).

---

## 3. Project background: DriveSense AI

**Name:** *DriveSense AI — Multi-Modal Automotive Perception & Risk Intelligence Platform* (repo folder `automotive-vision-pipeline`).

**Original research question:** *"Does adding real-time visual/environmental features to a crash-history model improve context-aware risk estimation?"*

### 3.1 Architecture (as in the README)
1. **Vision & perception engine** (OpenCV + YOLOv8 + ByteTrack): lane geometry (Canny + Hough → lateral offset in metres, curvature), object detection with persistent track IDs, pinhole distance and time-to-collision (TTC), scene photometry (traffic density, VRU ratio, glare index).
2. **Feature fusion:** 12-dim vision vector + 9-dim "crash history prior" vector → 21-dim fused vector.
3. **Models (XGBoost):** A = crash baseline (tabular), B = vision-only, C = fused. **TreeSHAP** for explanations with a Top-3 drivers display.
4. **Dashboard:** FastAPI backend + split-screen web UI (live stream with HUD overlays, risk gauge, SHAP factors, risk timeline).

### 3.2 Maths used (README)
- Pinhole distance: `d = f · H_class / h_bbox`. Assumed heights: car 1.50 m, truck 3.20 m, bus 3.40 m, motorcycle 1.20 m, pedestrian 1.70 m.
- Closing velocity: `(d_{t−Δt} − d_t)/Δt`; `TTC = d_t / closing_velocity` if closing velocity > 0.4 m/s, else 10.0 s.
- Lateral lane offset from lane-line pixel positions with assumed lane width 3.7 m and camera centred on the vehicle axis.

### 3.2b Repo layout (from README and screenshot)
```
automotive-vision-pipeline/
├── data/ docs/ models/ outputs/ scripts/ src/ web/
├── docs/        baseline_A_metrics.md, feature_schema.md, interview_qa_defense.md,
│                known_limitations_indian_roads.md, model_comparison_report.md, resume_bullets.md
├── experiments/
│   ├── exp1_crss/   (01_inspect_crss, 03_clean_and_merge, 04_train_models,
│   │                 05_evaluate_models, 06_shap_analysis + results/)
│   └── exp3_ccd/    (00_inspect_ccd, 00b_build_crash_table, 01_extract_features,
│                     02_build_video_features, 03_temporal_analysis, config.py, utils.py,
│                     perception/ package, README_exp3.md, requirements_exp3.txt,
│                     results/ and results_v1_backup/ with demo_videos, exp3_v2_test,
│                     figures, frame_features, ccd_inspection_summary.json, ccd_inventory.csv,
│                     exp3_summary.json, frame_features_all.csv, group_tests.csv,
│                     qa_report.json, temporal_tests.csv, video_features.csv, window_features.csv)
├── models/      feature_columns.pkl, model_a_baseline.json, model_b_vision.json,
│                model_c_fused.json, risk_model.pkl
├── src/         feature_schema, lane_detection, object_detection, ttc_distance, scene_features,
│                baseline_model_a, fusion_model, explainability, indian_road_audit, pipeline, dashboard
└── web/         index.html, index.css, app.js
```
Note: `exp3_ccd` has scripts `00`–`03` only — **no train/evaluate script yet** (no classifier evaluation on CCD). There is no `exp2` folder (KITTI distance validation not done).

---

## 4. The old (invalid) headline claim — for context only

README benchmark (Model A / B / C): ROC-AUC 0.724 / 0.815 / 0.8935; F1 0.6927 / 0.7662 / 0.8627; "+17.0% ROC-AUC lift and +24.5% F1 improvement." Two problems beyond the synthetic issue: the ROC-AUC "+17.0%" is actually **+0.17 absolute (≈ +23% relative)** while F1's 24.5% is **relative** — mixed conventions. Always report **absolute points** with confidence intervals.

Earlier honest self-assessment: all components are off-the-shelf (YOLOv8, ByteTrack, Canny/Hough, XGBoost, SHAP); benchmark came from a single video; no external baseline; no significance testing. Agreed framing: **an applied empirical/systems study, not a novel-algorithm paper.**

---

## 5. Critical audit finding (the reason the paper was re-scoped)

A code audit by Siddhi's side established:
- **Training data is fully synthetic.** `generate_multimodal_dataset()` draws 15,000 independent rows from `np.random`. Label: `latent = −3.2 + kinetic_threat + lane_threat + vru_threat + macro_threat + Normal(0, 0.35)`, `label = 1 if sigmoid(latent) > 0.45`. `kinetic_threat = 3.8·exp(−ttc/1.8) + 0.12·closing_vel` (dominant term); `lane_threat` fires on offset > 0.65 m or lane confidence < 0.45; `vru_threat` on `vru_ratio > 0.4` and density > 4; `macro_threat` from road surface, impairment, unlit darkness, `hist_risk`.
- The label is a formula over the same features the models train on, so **Model C beating A and B is built into the generator**; the "lift" measures the author-chosen weights, not real crashes.
- The "NHTSA CRSS" tabular features in the original training script were **hand-drawn with chosen probabilities** (and `hist_risk ~ Beta(2,10)`); nothing read real CRSS. Report text calling them NHTSA CRSS was inaccurate.
- At inference, `pipeline.py` used **one constant prior vector for every frame** (55 mph, straight, dry, daylight, clear, middle age, no impairment, car, 0.12). The dashboard `/predict` also used mostly fixed defaults.
- **No per-frame ground-truth label exists** for the demo video; the per-frame number is the model's predicted probability binned into 25/45/70% bands.
- Consequences: SHAP drivers reflect the generator weights; real frames are out-of-distribution (training daylight glare ≈ 0.008 vs ≈ 0.085 measured on a real frame; training TTC exact vs deployed TTC a noisy monocular estimate). `Conclusion: YES` was a hard-coded string in the report writer. Some vision features in the generator were partly derived from the tabular ones.

**What must never appear in the paper as a real-world result:** the A/B/C lift, "NHTSA CRSS" fusion on video, any SHAP drivers from the synthetic model, any claim that Model C "detects crashes."
**Acceptable:** a clearly labelled *synthetic benchmark pitfall* paragraph — how formula-derived labels inflate fusion gains — as a methodological lesson.

---

## 6. Decisions made so far (chronological)

1. Considered IRIS 27 vs ICAIDIET'26 (first-edition, SNS College of Technology + Yorkville University, Wiley/Scopus, deadline 21 Sep). Siddhi **dropped ICAIDIET** and focused on IRIS 27.
2. Drafted a fresh abstract for IRIS (deadline 15 Oct, 250–500 words, template in §2). Abstract was **selected**. *(The exact accepted abstract text is not in this file — ask her to paste it; the paper must stay consistent with it.)*
3. Realised during the audit that the lift was synthetic → re-scoped to **validate on real data**: real NHTSA CRSS for a tabular baseline; a real labelled crash dataset (CCD) for the vision half, with a TTC-threshold baseline.
4. Reframed contribution as an **empirical study + honest failure analysis** (not "our model wins").
5. Lost all local data once (inputs, outputs, dataset); only GitHub remains (code, docs, model JSONs). **Back up results now** (zip `results/`, push small CSV/JSON to GitHub, copy the rest to Drive).

---

## 7. Experiments and current status

| Exp | What | Status |
|---|---|---|
| **1** | NHTSA **CRSS 2024** severity baseline (real data) | **Done** |
| **2** | KITTI monocular distance validation (planned) | **Not done** — she could not get the complete dataset. *(Confirm exactly what Exp 2 was meant to be.)* Plan: cut, or use a small subset, or list as future work. |
| **3** | CCD (Car Crash Dataset) crash-anticipation study | **Partly done**: features extracted, temporal tests on **57 crash videos**. Missing: normal-clip control comparison and a classifier. |

### 7.1 Exp 1 — CRSS severity baseline (real)
- 50,540 crashes, 27 features. XGBoost test **ROC-AUC 0.813**; logistic regression 0.798; random forest 0.810.
- Scripts: `experiments/exp1_crss/01_inspect_crss.py … 06_shap_analysis.py`; there is SHAP analysis. *(Ask for exact target definition, split method, and SHAP top features before writing.)*
- Role in the paper: a clean, real-data tabular baseline ("what crash-history features alone can do"). Do not claim it is fused with the video.

### 7.2 Exp 3 — CCD dataset
- **CCD:** 1,500 crash + 3,000 normal dashcam clips with an official train/test split. Normal-clip features were downloaded and extraction was running as of the last notes; **unknown whether complete.**
- **Pipeline per clip** (frame-level → video-level → window-level): detection + tracking (YOLOv8 + ByteTrack), TTC/distance, lane detection (fails on most CCD clips — night, dew, rain, sunlight; no exact failure count computed), photometry, and **optical flow** (`flow_mag_center`, `flow_div_center`).
- **Temporal analysis design:** features computed in **early / mid / late windows aligned to the annotated accident time** (`alignment: "annotation"`), compared with Friedman and Wilcoxon signed-rank tests, **Holm-corrected** across 14 features, with bootstrap 95% CIs on the median difference (late − early).
- `exp3_summary.json`: 57 videos, all three windows available for all 57; exploratory clustering k = 2 with silhouette 0.333 (weak structure — skip or mention in one sentence).

#### Results (57 crash videos, early→late), from `temporal_tests.csv`
| Feature | n | Median early → late | Median diff (late−early) [95% CI] | % videos increase / decrease | Holm p |
|---|---|---|---|---|---|
| `has_det` (detector coverage) | 57 | 0.80 → 0.10 | −0.3 [−0.5, −0.2] | 7% / 68% | 1.0e-6 |
| `n_vehicles` | 57 | 1.1 → 0 | −1.0 [−1.4, −0.8] | 7% / 79% | 1.5e-7 |
| `occupancy` | 57 | 0.0093 → 0.00014 | −0.0090 [−0.0146, −0.0027] | 12% / 75% | 3.8e-7 |
| `has_ttc` | 57 | 0.5 → 0 | −0.2 [−0.4, 0] | 9% / 56% | 1.9e-5 |
| `flow_mag_center` | 57 | 0.0409 → 0.0793 | +0.0356 [0.0257, 0.0445] | **88% / 12%** | **5.6e-7** |
| `min_dist_path_m` | 21 | 13.9 → 24.7 m | +2.79 [1.54, 8.69] | 81% / 19% | 0.0022 (**suspicious**) |
| `max_inv_ttc_path` | 19 | 0.486 → 0.345 | −0.085 [−0.239, 0] | 26% / 68% | 0.38 (n.s., opposite of expected) |
| `sharpness` | 57 | 349.7 → 369.5 | +14.7 [5.9, 28.1] | 63% / 37% | 0.41 (n.s.) |
| `flow_div_center` | 57 | 0.254 → 0.314 | +0.033 [−0.046, 0.068] | 53% / 47% | 1.0 (n.s.) |
| `n_vru`, `brightness`, `contrast`, `glare_frac`, `dark_channel` | 57 | essentially flat | CIs include 0 | ~even | 0.91–1.0 (n.s.) |

**Weather group test** (`group_tests.csv`, Normal n=28 vs Adverse n=29, Mann-Whitney, Holm): no feature significant (all Holm p = 1.0) — low power; use as a limitation, not a finding. Photometry features (brightness, contrast, glare, dark channel) are flat → useful **negative controls** showing windows are not just lighting changes.

#### Interpretation (agreed headline direction)
> *Detection-based geometry (TTC, distance) loses coverage exactly as the crash approaches, while a detector-free signal — central optical-flow magnitude — rises sharply and consistently before the crash.*

- Strong, defensible findings: detector/tracker coverage collapse; optical-flow rise.
- Honest negative: TTC features are available in only 19 of 57 clips (33%), and inverse-TTC does **not** rise as expected after correction.
- **Likely artifacts to verify before claiming:** vehicle count falling to zero and path distance *increasing* near a crash may reflect truncated/oversized boxes (monocular distance overestimates when boxes are cut off), objects filling or leaving the frame, or detector failure at close range. Inspect 5–10 late-window frames before writing any causal explanation.

### 7.3 The critical gap: no control group
All tests compare windows **within crash clips only**, so the effects cannot yet be called crash-specific (camera shake, ego braking, clip structure could cause them). **Must-do:** apply identical relative windows to **normal** CCD clips and compare late−early changes, crash vs normal (Mann-Whitney + Holm, with effect sizes and CIs). If the effect is not crash-specific, the honest result becomes "these changes are generic to clip structure" — still reportable, but a different paper; better to find out now.

### 7.4 Strongly recommended addition: real-label classifier
Clip-level classifier (logistic regression and/or XGBoost) on window features (flow + detection coverage + TTC + photometry) vs a **TTC-threshold baseline** (threshold tuned on a validation split only), on the **official CCD train/test split, split by clip never by frame**, fixed seeds, bootstrap CIs, AUC/AP. If CCD gives accident timestamps, also report earliness (time-to-accident). Add feature-group ablation and SHAP on this **real-label** model.

---

## 8. Demo recording observation (`Screen_Recording_2026-11-21_113451.mp4`)
A 9-second screen recording of the DriveSense dashboard on a dense Indian-road video (trucks, motorbikes, pedestrians). Model C shows **96–97% crash probability** nearly throughout, TTC 0.2–0.5 s while boxes are 5–20 m away and many tracks are labelled "STABLE", and **closing velocity pinned at exactly 40.0 m/s** in every frame inspected (looks like a clipped cap); risk trajectory saturated/spiky. This is the synthetic-trained model failing out of distribution.
**Use:** a cheap real diagnostic for the failure analysis — measure on that clip how often TTC is available and how often closing velocity is clipped. **Do not** present it as validation, and do not use its frames as figures (third-party video → permission forms).

---

## 9. What goes in the paper (recommended plan)

**Working title direction:** *Do Monocular Perception Features Carry Pre-Crash Signal? An Interpretable Pipeline and Empirical Study on CCD* (or similar; must stay consistent with the accepted abstract).

**Contributions (3):**
1. Open, interpretable monocular perception pipeline (YOLOv8 + ByteTrack, geometry/TTC, optical flow, photometry) with SHAP explanations.
2. A real-data temporal study on CCD with statistical tests, **plus normal-clip control** — headline: detector-based features lose coverage while optical flow rises.
3. A quantified failure analysis: TTC availability, lane-detector failure rates by condition (night/rain/dew/glare), closing-velocity clipping, dense Indian traffic.

**Supporting:** Exp 1 CRSS real-data baseline; classifier vs TTC-threshold baseline (§7.4); a short **"synthetic-benchmark pitfall"** paragraph; fusion of real CRSS priors with video listed as **future work**.

**Suggested 10-page budget:** Intro 0.8 · Related work 0.7 · Pipeline 1.5 · Datasets & protocol 1.0 · Results — CRSS 1.0, CCD temporal with control 2.0, classifier vs baseline 1.0, failure analysis 1.0 · Limitations & future work 0.5 · Conclusion 0.3 · References ~1.0 (≈ 10–11; trim to fit).

**Figures/tables (all her own, B&W-legible):** pipeline diagram; mean ± CI curves of flow and detection coverage around the accident time (crash vs normal); detection-coverage bar chart; ROC/PR vs TTC baseline; SHAP bar chart; one results table with effect sizes and Holm p-values; a dataset/protocol table.

**Skip/downplay:** CRSS-video fusion claims (unless metadata mapping is trivial), KITTI distance validation (limitation instead), clustering, the dashboard/UI. UI polish is irrelevant to a B&W Springer print.

**Suggested references to verify and cite (not yet checked):** CCD (Bao et al., ACM MM 2020), DAD (Chan et al., 2016), DoTA (Yao et al.), KITTI (Geiger et al., 2012), YOLOv8/Ultralytics, ByteTrack (Zhang et al., ECCV 2022), XGBoost (Chen & Guestrin, 2016), SHAP/TreeSHAP (Lundberg & Lee, 2017; Lundberg et al., 2020), NHTSA CRSS documentation, the specific optical-flow method actually used (e.g., Farnebäck). Confirm each reference's details before use.

---

## 10. Schedule to the deadline

| Day | Task |
|---|---|
| **Sat 21 Nov** | Back up everything. Start normal-clip extraction (overnight). Download Springer template; write skeleton + related work. Decide author list and corresponding author. |
| **Sun 22** | Control comparison (crash vs normal windows); classifier vs TTC baseline with CIs; final figures; **freeze results**. Inspect artifact clips. |
| **Mon 23** | Full draft in the Springer template; check 10-page limit, references, cross-refs. |
| **Tue 24** | Proofread; compile final PDF; fill, sign and scan the Contributor Agreement; assemble package; draft submission email. |
| **Wed 25 (morning)** | Send. Do not wait for the evening. |

**Fallbacks:** if the classifier slips → the temporal study with control plus the failure analysis is still a coherent paper. If the control is not feasible → report within-crash effects explicitly as *within-crash* with the limitation stated up front.

---

## 11. Honest quality assessment (so expectations stay calibrated)
- Sufficient for a Springer LNEE conference paper at IRIS if framed as an applied empirical study with honest scope: real data (CRSS, CCD), proper statistics (Holm correction, CIs, clip-level splits), negative controls, and a candid failure analysis.
- Realistically a solid, honest paper, not a novelty-driven best-paper candidate; polished execution can still do well in the presentation.
- Weak points reviewers will probe: small n (57 crash clips — scale up if extraction speed allows), no control group (§7.3), unvalidated monocular distance/TTC, brittle KITTI-tuned lane detector, off-the-shelf components, missing Exp 2.
- Springer will re-review and run plagiarism checks; keep the paper consistent with the accepted abstract; any deviation in results should be explained honestly.

---

## 12. Housekeeping checklist
- [ ] Results backed up (GitHub for small files, Drive for large).
- [ ] One single reproducible run (resolve `results/` vs `results_v1_backup/` vs `exp3_v2_test`); every number traceable to a script + file; `qa_report.json` consistent.
- [ ] README / `model_comparison_report.md` reworded: synthetic benchmark labelled as such; no "NHTSA CRSS" claim on synthetic data; remove hard-coded `Conclusion: YES`.
- [ ] Author list, order, affiliations (IIIT Nagpur, city, country), corresponding author + email fixed.
- [ ] Springer template downloaded (EE/physics/maths package; not LNCS).
- [ ] No third-party frames/figures in the paper (or permissions obtained and checklist filled).
- [ ] Contributor Agreement signed by the corresponding author, scanned.
- [ ] PDF matches source exactly; exactly 10 pages; references in order, DOIs, no hyperlinks.
- [ ] Email sent to `conference@shaastra.org` with the required subject line, before the evening of 25 Nov.
- [ ] Later (after submission): presentation prep for 2–3 Jan 2027 (oral/poster).

---

## 13. Quick answers to likely questions
- **Is the old +17% lift usable?** No (synthetic). Use only as a labelled pitfall example.
- **Can I include the dashboard?** Not in the paper; optionally as a demo for the talk, with a "synthetic demo model" label.
- **Do I need the permission/release forms?** Only if third-party content appears. Avoid by using only her own figures.
- **What if the control shows no crash-specific effect?** Report honestly; reframe as "temporal changes in dashcam features within CCD crash clips are/aren't specific…" and keep the failure analysis as the main contribution.

---

## 14. Unknowns to confirm with Siddhi first
1. The **accepted abstract text** and title (paper must match it).
2. **Author list / order / corresponding author** and email(s).
3. Whether she writes in **Word or LaTeX** (and which Springer template she downloaded).
4. What **Exp 2** was exactly and how much of it exists; whether to cut or keep as a limitation.
5. Whether **normal-clip features are fully extracted** and how many crash clips can be processed by Sunday.
6. How **windows** (early/mid/late) are defined (seconds/frames before the annotated accident) and how optical flow is computed (method, resolution).
7. Exp 1 details: target definition, split, top SHAP features.
8. Whether a **classifier evaluation** script exists for CCD (none visible in the folder).
9. Which run (v1 backup vs v2 vs current) is the canonical one, and the status of `qa_report.json`.
10. Hardware/time constraints for overnight extraction.
