#!/usr/bin/env python3
"""
select_triptych_candidates_stage1.py

Stage 1 (local, cheap) of triptych candidate selection for Adam's
inversion-fidelity gate. Narrows the full 8,407-image labeled EyePACS pool
(eyepacs_from_zip/) down to a manageable shortlist before spending GPU time
running the learned EX/HE detectors on every image (Stage 2, pod).

Filters:
  1. DR grade 2-4 (moderate NPDR through PDR) -- grade 1 is often just a
     handful of microaneurysms, too subtle for an "unambiguous" pathology
     demo; grade 0 has no pathology at all.
  2. Basic exposure quality -- reject images that are too dark or blown out
     (mean L-channel brightness outside a sane range), since Adam's ask is
     specifically for "well-exposed" images, and detector confidence means
     less on a badly-exposed photo.

Output: triptych_candidates_stage1.csv (stem, grade, mean_brightness),
capped at --max_pool images (default 300) so Stage 2's detector inference
stays cheap. Stratified across grades 2/3/4 rather than taking the largest
grade's images only.

Run (local):
    conda run -n py311 python select_triptych_candidates_stage1.py
"""
import argparse
import random
import cv2
import numpy as np
from pathlib import Path

RETINA = Path(r"C:\Users\nyama\Retina_Project")
LABELS_CSV = RETINA / "trainLabels.csv"
SAMPLE_DIR = RETINA / "eyepacs_from_zip"
OUT_CSV = RETINA / "triptych_candidates_stage1.csv"

MIN_BRIGHTNESS = 40   # L-channel mean, 0-255 -- reject near-black
MAX_BRIGHTNESS = 200  # reject blown-out/overexposed
GRADES = [2, 3, 4]


def brightness_ok(img_path: Path) -> tuple[bool, float]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return False, 0.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    mean_l = float(lab[:, :, 0].mean())
    return (MIN_BRIGHTNESS <= mean_l <= MAX_BRIGHTNESS), mean_l


def main(args):
    available = {p.stem for p in SAMPLE_DIR.glob("*.jpeg")}
    print(f"Local EyePACS pool: {len(available)} images")

    by_grade = {g: [] for g in GRADES}
    with open(LABELS_CSV) as f:
        next(f)
        for line in f:
            stem, level = line.strip().split(",")
            level = int(level)
            if stem in available and level in by_grade:
                by_grade[level].append(stem)

    for g in GRADES:
        print(f"  grade {g}: {len(by_grade[g])} available locally")

    random.seed(42)
    per_grade_cap = args.max_pool // len(GRADES)
    shortlist = []
    for g in GRADES:
        pool = by_grade[g]
        random.shuffle(pool)
        shortlist += [(s, g) for s in pool[: per_grade_cap * 3]]  # oversample before quality filter

    print(f"\nChecking exposure quality on {len(shortlist)} candidates...")
    kept = []
    for stem, grade in shortlist:
        ok, brightness = brightness_ok(SAMPLE_DIR / f"{stem}.jpeg")
        if ok:
            kept.append((stem, grade, brightness))

    # Cap per grade after quality filter, keep it balanced
    final = []
    for g in GRADES:
        grade_kept = [k for k in kept if k[1] == g][:per_grade_cap]
        final += grade_kept

    print(f"Kept after exposure filter: {len(kept)} -> final shortlist: {len(final)}")

    with open(OUT_CSV, "w") as f:
        f.write("stem,grade,mean_brightness\n")
        for stem, grade, brightness in final:
            f.write(f"{stem},{grade},{brightness:.1f}\n")
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_pool", type=int, default=300,
                    help="Target shortlist size before Stage 2 detector screening")
    main(p.parse_args())
