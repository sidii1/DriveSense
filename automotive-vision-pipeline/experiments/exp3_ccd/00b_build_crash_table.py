"""STEP 0b: convert Crash-1500.txt -> Crash_Table.csv (with derived crash onset frame) next to the videos.

Line format (verified on the first lines of the real file):
  video_id,[50 per-frame labels 0/1],srcfield_a,srcfield_b,lighting,weather,ego_involved
Crash onset = index of the first frame labelled 1.

python 00b_build_crash_table.py [path\\to\\Crash-1500.txt]
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import config as C


def find_txt():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for base in (C.CCD_DIR, C.CCD_DIR.parent):
        hits = list(base.glob("Crash-1500*.txt")) + list(base.rglob("Crash-1500*.txt"))
        if hits:
            return hits[0]
    raise FileNotFoundError("Crash-1500.txt not found; pass its path as an argument")


def main():
    txt = find_txt()
    print("reading", txt)
    rows, bad = [], 0
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^,]+),\s*\[([^\]]*)\]\s*,?(.*)$", line)
        if not m:
            bad += 1
            continue
        vid, lab, rest = m.group(1).strip(), m.group(2), m.group(3)
        labels = [int(x) for x in lab.replace(" ", "").split(",") if x != ""]
        f = [x.strip() for x in rest.split(",")]
        if len(f) < 3:
            bad += 1
            continue
        onset = int(np.argmax(np.array(labels) == 1)) if 1 in labels else np.nan
        rows.append(dict(vidname=vid, crash_frame=onset, n_labels=len(labels),
                         labels_monotone=bool(all(b >= a for a, b in zip(labels, labels[1:]))),
                         src_a=f[0] if len(f) >= 5 else "", src_b=f[1] if len(f) >= 5 else "",
                         timing=f[-3], weather=f[-2], egoinvolve=f[-1]))
    df = pd.DataFrame(rows)
    out = C.CCD_DIR / "Crash_Table.csv"
    df.to_csv(out, index=False)
    print(f"parsed {len(df)} rows ({bad} unparsable) -> {out}")
    print("label length:", df.n_labels.value_counts().to_dict(), "| monotone 0->1:", df.labels_monotone.mean())
    print("onset frame: min", df.crash_frame.min(), "max", df.crash_frame.max(),
          "| missing onset:", int(df.crash_frame.isna().sum()))
    print("onset (s) quantiles:", (df.crash_frame / 10).quantile([0, .25, .5, .75, 1]).round(2).to_dict())
    for c in ["timing", "weather", "egoinvolve"]:
        print(f"\n{c}:\n{df[c].value_counts(dropna=False).to_string()}")


if __name__ == "__main__":
    main()
