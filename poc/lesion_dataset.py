"""
IDRiD lesion dataset for disease-subspace supervision.

Loads real fundus images paired with:
  - per-pixel HE / EX / MA segmentation masks  (from IDRiD A. Segmentation)
  - DR retinopathy grade  0-4                   (from IDRiD B. Disease Grading)
  - risk of macular edema  0-2                  (from IDRiD B. Disease Grading)

Each sample dict:
  image      : np.float32 (H, W, 3)  RGB, CLAHE-preprocessed, [0, 1]
  he_mask    : np.float32 (H, W)     binary HE  pixel mask
  ex_mask    : np.float32 (H, W)     binary EX  pixel mask
  ma_mask    : np.float32 (H, W)     binary MA  pixel mask
  dr_grade   : int  0-4   (-1 if CSV not provided)
  me_risk    : int  0-2   (-1 if CSV not provided)
  patient_id : str  e.g. "IDRiD_01"

Usage:
    from realizemd_deid.poc.lesion_dataset import IDRiDLesionDataset

    ds = IDRiDLesionDataset(
        seg_dir  = "A. Segmentation/A. Segmentation",
        grade_csv= "idrid_grading/B. Disease Grading/2. Groundtruths/"
                   "a. IDRiD_Disease Grading_Training Labels.csv",
        split    = "train",   # "train" or "test"
    )
    sample = ds[0]
    print(sample["patient_id"], sample["dr_grade"],
          sample["he_mask"].sum(), sample["ex_mask"].sum())
"""
import csv
from pathlib import Path

import cv2
import numpy as np


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L channel — matches train_he_detector.py preprocessing exactly."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return (rgb / 255.0).astype(np.float32)


# ── Grade CSV loader ───────────────────────────────────────────────────────────

def _load_grades(csv_path: Path) -> dict:
    """Returns {patient_id_3digit: (dr_grade, me_risk)} e.g. {"IDRiD_001": (3, 2)}."""
    grades = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Image name"].strip()
            try:
                dr   = int(row["Retinopathy grade"].strip())
                me   = int(row["Risk of macular edema "].strip())
            except (ValueError, KeyError):
                dr, me = -1, -1
            grades[name] = (dr, me)
    return grades


def _seg_id_to_grade_id(seg_id: str) -> str:
    """IDRiD_01 → IDRiD_001  (segmentation uses 2-digit, grading uses 3-digit)."""
    prefix, num = seg_id.rsplit("_", 1)
    return f"{prefix}_{int(num):03d}"


# ── Dataset ───────────────────────────────────────────────────────────────────

class IDRiDLesionDataset:
    """
    Args:
        seg_dir   : path to IDRiD "A. Segmentation" root folder
        grade_csv : path to IDRiD training or testing labels CSV (optional)
        split     : "train" or "test"
    """

    _SPLIT_MAP = {"train": "a. Training Set", "test": "b. Testing Set"}
    _MASK_DIRS = {
        "he": "2. Haemorrhages",
        "ex": "3. Hard Exudates",
        "ma": "1. Microaneurysms",
    }
    _MASK_SUFFIX = {"he": "_HE", "ex": "_EX", "ma": "_MA"}

    def __init__(self, seg_dir: str, grade_csv: str = None, split: str = "train"):
        self.seg_root = Path(seg_dir)
        self.split    = split
        split_folder  = self._SPLIT_MAP[split]

        self.img_dir  = self.seg_root / "1. Original Images" / split_folder
        self.gt_root  = self.seg_root / "2. All Segmentation Groundtruths" / split_folder

        self.grades = _load_grades(Path(grade_csv)) if grade_csv else {}

        # Build sample list from images present
        self.samples = sorted(
            p.stem for p in self.img_dir.glob("*.jpg")
        )
        assert len(self.samples) > 0, f"No images found in {self.img_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        patient_id = self.samples[idx]

        # Image
        img_path = self.img_dir / f"{patient_id}.jpg"
        img_bgr  = cv2.imread(str(img_path))
        assert img_bgr is not None, f"Could not read image: {img_path}"
        image = preprocess_image(img_bgr)

        # Masks — missing mask → all-zero (image has no lesions of that type)
        masks = {}
        for key, subdir in self._MASK_DIRS.items():
            suffix   = self._MASK_SUFFIX[key]
            mask_dir = self.gt_root / subdir
            # try .tif first, then .png
            mask_path = next(
                (mask_dir / f"{patient_id}{suffix}{ext}"
                 for ext in (".tif", ".png", ".bmp")
                 if (mask_dir / f"{patient_id}{suffix}{ext}").exists()),
                None
            )
            if mask_path is not None:
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                masks[key] = (m > 0).astype(np.float32) if m is not None else \
                             np.zeros(image.shape[:2], dtype=np.float32)
            else:
                masks[key] = np.zeros(image.shape[:2], dtype=np.float32)

        # DR grade
        grade_id = _seg_id_to_grade_id(patient_id)
        dr_grade, me_risk = self.grades.get(grade_id, (-1, -1))

        return dict(
            image      = image,
            he_mask    = masks["he"],
            ex_mask    = masks["ex"],
            ma_mask    = masks["ma"],
            dr_grade   = dr_grade,
            me_risk    = me_risk,
            patient_id = patient_id,
        )

    def summary(self):
        """Print dataset statistics — call after loading to verify."""
        he_count = ex_count = ma_count = 0
        grades = []
        for i in range(len(self)):
            s = self[i]
            if s["he_mask"].sum() > 0: he_count += 1
            if s["ex_mask"].sum() > 0: ex_count += 1
            if s["ma_mask"].sum() > 0: ma_count += 1
            if s["dr_grade"] >= 0: grades.append(s["dr_grade"])
        print(f"IDRiDLesionDataset ({self.split}): {len(self)} images")
        print(f"  HE masks present : {he_count}/{len(self)}")
        print(f"  EX masks present : {ex_count}/{len(self)}")
        print(f"  MA masks present : {ma_count}/{len(self)}")
        if grades:
            from collections import Counter
            dist = Counter(grades)
            print(f"  DR grade dist   : { {k: dist[k] for k in sorted(dist)} }")
        else:
            print(f"  DR grades       : not loaded")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    seg_dir   = sys.argv[1] if len(sys.argv) > 1 else \
                r"A. Segmentation\A. Segmentation"
    grade_csv = sys.argv[2] if len(sys.argv) > 2 else \
                r"idrid_grading\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"

    ds = IDRiDLesionDataset(seg_dir=seg_dir, grade_csv=grade_csv, split="train")
    ds.summary()
    s = ds[0]
    print(f"\nSample 0: {s['patient_id']}  dr={s['dr_grade']}  me={s['me_risk']}")
    print(f"  image shape : {s['image'].shape}  range [{s['image'].min():.2f}, {s['image'].max():.2f}]")
    print(f"  he_mask px  : {int(s['he_mask'].sum())}")
    print(f"  ex_mask px  : {int(s['ex_mask'].sum())}")
    print(f"  ma_mask px  : {int(s['ma_mask'].sum())}")
