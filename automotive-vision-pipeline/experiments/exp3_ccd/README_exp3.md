# Experiment 3 - Crash-scene perception and temporal risk indicators (CCD Crash-1500)

Place under `experiments/exp3_ccd/`; put Crash-1500 videos + Crash_Table.csv in `data/raw/ccd/`
(or set EXP3_CCD_DIR).  Run from inside this folder:

    pip install -r requirements_exp3.txt
    python 00_inspect_ccd.py                       # decides what claims are possible
    python 01_extract_features.py --limit 10 --demo 3   # smoke test, check results/demo_videos
    python 01_extract_features.py --demo 5         # full run, resumable
    python 02_build_video_features.py              # QA + window tables
    python 03_temporal_analysis.py                 # stats + figures

Outputs land in results/: frame_features_all.csv, window_features.csv, video_features.csv,
qa_report.json, temporal_tests.csv, group_tests.csv, cluster_profiles.csv, figures/, demo_videos/.
