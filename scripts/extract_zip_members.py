"""extract_zip_members.py -- pull named files out of a partial (first-piece-only) split zip.

Same method as extract_partial_zip.py (Retina_Project root), which built the local `eyepacs_from_zip/`
from gdrive:retinal_vessel_segmentation_project/eyepacs/train.zip.001: that piece has no central
directory, so local file headers (PK\\x03\\x04) are scanned and each fully contained entry is read
using the size in its header. This version extracts only the names listed in --names (one per line),
used to assemble the 450 healthy photos for notebooks/detector_evaluation.ipynb on RunPod.
The notebook then verifies every extracted file against the MD5 recorded from the local originals.

Usage:
    python scripts/extract_zip_members.py --zip train.zip.001 --names eyepacs_selected.txt --out eyepacs_from_zip
"""
import argparse, mmap, struct, zlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--names", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
want = {l.strip() for l in open(a.names) if l.strip()}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
size, pos, got = Path(a.zip).stat().st_size, 0, set()
with open(a.zip, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        while len(got) < len(want):
            idx = mm.find(b"PK\x03\x04", pos)
            if idx == -1 or idx + 30 > size:
                break
            h = mm[idx:idx + 30]
            gp, meth, csize = struct.unpack("<H", h[6:8])[0], struct.unpack("<H", h[8:10])[0], struct.unpack("<I", h[18:22])[0]
            nlen, elen = struct.unpack("<H", h[26:28])[0], struct.unpack("<H", h[28:30])[0]
            name = mm[idx + 30:idx + 30 + nlen].decode("utf-8", errors="replace")
            start = idx + 30 + nlen + elen; end = start + csize
            if gp & 0x08 or meth not in (0, 8):
                pos = idx + 4; continue
            if end > size:
                break
            base = Path(name).name
            if base in want:
                raw = mm[start:end]
                if meth == 8:
                    raw = zlib.decompressobj(-15).decompress(raw)
                (out / base).write_bytes(raw); got.add(base)
            pos = end
    finally:
        mm.close()
missing = sorted(want - got)
print(f"extracted {len(got)} of {len(want)} into {out}" + (f"; MISSING {len(missing)}: {missing[:5]}" if missing else ""))
raise SystemExit(1 if missing else 0)
