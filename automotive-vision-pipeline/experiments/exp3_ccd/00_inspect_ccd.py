"""STEP 0: inspect Crash-1500 videos + annotations. Decides whether pre-crash timing is available."""
import json
import cv2
import pandas as pd
import config as C
from utils import list_videos, norm_key, find_col, VIDEO_EXT


def main():
    vids = list_videos()
    print(f"Videos found under {C.CCD_DIR}: {len(vids)}")
    if not vids:
        print("No video files found. If CCD came as folders of .jpg frames, tell me and I will add a frame-folder reader.")
        return
    rows = []
    for p in vids:
        cap = cv2.VideoCapture(str(p))
        rows.append(dict(video_id=p.stem, path=str(p.relative_to(C.CCD_DIR)),
                         n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         fps=round(cap.get(cv2.CAP_PROP_FPS), 2),
                         width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
    inv = pd.DataFrame(rows)
    inv.to_csv(C.OUT / "ccd_inventory.csv", index=False)
    print("\nframe-count distribution:\n", inv.n_frames.value_counts().head(8))
    print("\nfps distribution:\n", inv.fps.value_counts().head(5))
    print("\nresolution distribution:\n", inv.groupby(["width", "height"]).size().head(5))

    summary = {"n_videos": len(inv), "n_frames_modal": int(inv.n_frames.mode()[0]),
               "fps_modal": float(inv.fps.mode()[0])}

    csvs = [p for p in C.CCD_DIR.rglob("*.csv")]
    print(f"\nMetadata CSVs: {[str(p.relative_to(C.CCD_DIR)) for p in csvs]}")
    for p in csvs:
        df = pd.read_csv(p)
        print(f"\n=== {p.name}  shape={df.shape}\ncolumns: {list(df.columns)}\n{df.head(3).to_string()}")
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique() <= 12:
                print(f"\n{c} value counts:\n{df[c].value_counts(dropna=False).to_string()}")
            elif pd.api.types.is_numeric_dtype(df[c]):
                print(f"{c}: min={df[c].min()}, max={df[c].max()}, nulls={df[c].isna().sum()}")
        vcol = find_col(df, ["vidname", "video", "name", "id"])
        if vcol is not None:
            keys = set(inv.video_id.map(norm_key))
            match = df[vcol].map(norm_key).isin(keys).mean()
            print(f"\nid column '{vcol}': {match:.1%} of metadata rows match a video file")
            summary["id_match_rate"] = float(match)
        ecol = find_col(df, ["startframe", "start_frame", "accident_frame", "crash_frame", "event"])
        print(f"\nCandidate event-timing column: {ecol}")
        summary["event_col"] = ecol
        for k in ["ego", "timing", "weather", "pattern"]:
            summary[f"{k}_col"] = find_col(df, [k])
    print("\nDECISION:",
          "event timing column found -> check its min/max above (should lie inside 0..n_frames) -> event-aligned analysis is possible"
          if summary.get("event_col") else
          "NO timing column -> only clip-end-assumption alignment is possible; say so in the paper")
    (C.OUT / "ccd_inspection_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
