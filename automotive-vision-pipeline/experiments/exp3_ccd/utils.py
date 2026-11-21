import re
from pathlib import Path
import config as C

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def list_videos(root=None):
    root = Path(root or C.CCD_DIR)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)


def norm_key(s):
    """'000123.mp4' / 123 / '123' -> '123' so metadata ids match file names."""
    s = str(s)
    s = re.sub(r"\.(mp4|avi|mov|mkv)$", "", s, flags=re.I)
    digits = re.sub(r"\D", "", s)
    return (digits.lstrip("0") or "0") if digits else s


def find_col(df, keys):
    for k in keys:
        for c in df.columns:
            if k in str(c).lower():
                return c
    return None


def find_metadata_csv():
    cands = [p for p in C.CCD_DIR.rglob("*.csv")]
    pref = [p for p in cands if "crash" in p.name.lower() or "table" in p.name.lower()]
    return (pref or cands or [None])[0]
