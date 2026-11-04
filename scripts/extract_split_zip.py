"""extract_split_zip.py -- extract members of a byte-split zip without joining the parts on disk.

gdrive:ddr_raw/DDR-dataset.zip.001 ... .010 are plain byte slices of one zip ("byte-split", decision.md
2026-10-10). Joining them with `cat` needs a second full-size copy (~15 GB); this script instead presents
the parts to Python's zipfile as one seekable stream and extracts only the members under --prefix.
Used on RunPod to get DDR lesion_segmentation/test for notebooks/detector_evaluation.ipynb, which then
verifies every extracted file against MD5s recorded from the local originals.

Usage:
    python scripts/extract_split_zip.py --parts /workspace/ddr_raw/DDR-dataset.zip.0?? \
        --prefix DDR-dataset/lesion_segmentation/test/ --out /workspace/detector_eval_pkg/scratch/ddr_lesion
"""
import argparse, bisect, io, os, zipfile


class Concat(io.RawIOBase):
    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.starts = [0]
        for p in paths:
            self.starts.append(self.starts[-1] + os.path.getsize(p))
        self.size, self.pos = self.starts[-1], 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = 0
        while n < len(b) and self.pos < self.size:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            f = self.files[i]; f.seek(self.pos - self.starts[i])
            chunk = f.read(min(len(b) - n, self.starts[i + 1] - self.pos))
            if not chunk:
                break
            b[n:n + len(chunk)] = chunk; n += len(chunk); self.pos += len(chunk)
        return n


ap = argparse.ArgumentParser()
ap.add_argument("--parts", nargs="+", required=True); ap.add_argument("--prefix", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
parts = sorted(a.parts)
print("parts:", [os.path.basename(p) for p in parts])
with zipfile.ZipFile(io.BufferedReader(Concat(parts), 1 << 20)) as z:
    members = [m for m in z.namelist() if m.startswith(a.prefix) and not m.endswith("/")]
    if not members:
        roots = sorted({m.split("/")[0] for m in z.namelist()})[:10]
        raise SystemExit(f"no members under {a.prefix!r}; top-level entries: {roots}")
    for m in members:
        z.extract(m, a.out)
print(f"extracted {len(members)} files under {a.prefix} into {a.out}")
