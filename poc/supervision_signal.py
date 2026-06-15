"""
supervision_signal.py
Defines and generates the disease-subspace supervision signal for the Path B
generative de-identification pipeline.

FORMAT SPEC (for Harriet's integration):
─────────────────────────────────────────
Each image produces a dict:

{
    "image_id"  : str          — IDRiD stem or EyePACS patient ID
    "dr_grade"  : int          — 0–4 (IDRiD grade; -1 if unavailable)
    "shape"     : (H, W)       — full-resolution image dimensions
    "masks": {
        "HE" : np.ndarray      — (H, W) uint8, 0/255  haemorrhages
        "EX" : np.ndarray      — (H, W) uint8, 0/255  hard exudates
        "MA" : np.ndarray      — (H, W) uint8, 0/255  microaneurysms
    },
    "combined"  : np.ndarray   — (H, W) uint8, 0/255  union of all types
    "has_lesion": bool         — True if any lesion present (convenience flag)
}

DESIGN NOTES:
- Masks are kept per-type so each lesion type can have its own loss weight
  in the perceptual lesion loss (HE vs EX vs MA may need different weighting).
- uint8 0/255 (not bool) for direct use as OpenCV masks and loss weight maps.
- HE uses the trained detector (requires checkpoint). EX/MA are rule-based.
- If a GT mask file exists for an image, it is used directly instead of the
  detector — GT is always preferred for training supervision.

INTERFACE WITH HARRIET:
  The `masks` dict feeds the perceptual lesion loss in the disease subspace:
    loss_lesion = sum over types: w_t * perceptual_loss(pred_t, gt_mask_t)
  Harriet defines the loss; Victoria provides the masks in this format.
  Downstream: supervision_signal.generate() is the single entry point.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.pathology import (
    detect_hard_exudates,
    detect_microaneurysms,
    detect_optic_disc,
)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def _load_gt_mask(gt_dir, stem, suffix):
    """Return GT mask as uint8 (0/255) or None if unavailable."""
    if gt_dir is None:
        return None
    for ext in (".tif", ".png"):
        p = Path(gt_dir) / f"{stem}{suffix}{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.max() > 0:
                return (m > 0).astype(np.uint8) * 255
    return None


def generate(
    img_bgr: np.ndarray,
    image_id: str,
    dr_grade: int = -1,
    gt_he_dir=None,
    gt_ex_dir=None,
    gt_ma_dir=None,
    he_model=None,
    he_threshold: float = 0.30,
    device: str = "cpu",
) -> dict:
    """
    Generate the supervision signal for one image.

    Args:
        img_bgr    : BGR image as loaded by cv2.imread
        image_id   : filename stem (e.g. "IDRiD_02")
        dr_grade   : diabetic retinopathy grade 0–4 (-1 if unknown)
        gt_he_dir  : folder of GT HE masks (preferred over detector if present)
        gt_ex_dir  : folder of GT EX masks
        gt_ma_dir  : folder of GT MA masks
        he_model   : trained HE detector (smp.Unet); if None, HE mask is zeros
        he_threshold: probability threshold for HE detector output
        device     : "cuda" or "cpu"

    Returns:
        Supervision signal dict — see module docstring for format.
    """
    img_rgb = preprocess(img_bgr)
    H, W    = img_rgb.shape[:2]
    od      = detect_optic_disc(img_rgb)

    # ── HE ────────────────────────────────────────────────────────────────────
    gt_he = _load_gt_mask(gt_he_dir, image_id, "_HE")
    if gt_he is not None:
        he_mask = gt_he
    elif he_model is not None:
        from poc.lesion_preservation_eval import get_prob_map
        prob    = get_prob_map(he_model, img_rgb, device=device)
        he_mask = ((prob > he_threshold) * 255).astype(np.uint8)
    else:
        he_mask = np.zeros((H, W), dtype=np.uint8)

    # ── EX ────────────────────────────────────────────────────────────────────
    gt_ex = _load_gt_mask(gt_ex_dir, image_id, "_EX")
    ex_mask = gt_ex if gt_ex is not None else detect_hard_exudates(img_rgb, od)

    # ── MA ────────────────────────────────────────────────────────────────────
    gt_ma = _load_gt_mask(gt_ma_dir, image_id, "_MA")
    ma_mask = gt_ma if gt_ma is not None else detect_microaneurysms(img_rgb, od)

    combined  = cv2.bitwise_or(cv2.bitwise_or(he_mask, ex_mask), ma_mask)
    has_lesion = combined.max() > 0

    return {
        "image_id"  : image_id,
        "dr_grade"  : dr_grade,
        "shape"     : (H, W),
        "masks"     : {"HE": he_mask, "EX": ex_mask, "MA": ma_mask},
        "combined"  : combined,
        "has_lesion": has_lesion,
    }


def generate_dataset(
    img_dir: str,
    out_dir: str,
    grade_csv: str = None,
    gt_he_dir: str = None,
    gt_ex_dir: str = None,
    gt_ma_dir: str = None,
    he_model=None,
    he_threshold: float = 0.30,
    device: str = "cpu",
):
    """
    Run generate() over a full image directory. Saves per-image mask .npz files.

    Output per image: {out_dir}/{stem}_supervision.npz
      keys: he_mask, ex_mask, ma_mask, combined, dr_grade, has_lesion
    """
    import pandas as pd

    grades = {}
    if grade_csv:
        df = pd.read_csv(grade_csv)
        # Expects columns: image (stem), level (0-4)
        grades = dict(zip(df["image"].astype(str), df["level"].astype(int)))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))

    for img_path in img_paths:
        stem    = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: could not read {img_path.name}")
            continue

        sig = generate(
            img_bgr, stem,
            dr_grade   = grades.get(stem, -1),
            gt_he_dir  = gt_he_dir,
            gt_ex_dir  = gt_ex_dir,
            gt_ma_dir  = gt_ma_dir,
            he_model   = he_model,
            he_threshold = he_threshold,
            device     = device,
        )

        np.savez_compressed(
            str(out / f"{stem}_supervision.npz"),
            he_mask    = sig["masks"]["HE"],
            ex_mask    = sig["masks"]["EX"],
            ma_mask    = sig["masks"]["MA"],
            combined   = sig["combined"],
            dr_grade   = np.int8(sig["dr_grade"]),
            has_lesion = np.bool_(sig["has_lesion"]),
        )
        status = "lesion" if sig["has_lesion"] else "clean"
        print(f"  {stem}  grade={sig['dr_grade']}  {status}")

    print(f"\nDone. {len(img_paths)} images → {out}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",    required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--grade_csv",  default=None, help="CSV with columns: image, level")
    p.add_argument("--gt_he_dir",  default=None)
    p.add_argument("--gt_ex_dir",  default=None)
    p.add_argument("--gt_ma_dir",  default=None)
    p.add_argument("--ckpt",       default=None, help="HE detector .pth (optional)")
    p.add_argument("--encoder",    default="efficientnet-b2")
    p.add_argument("--threshold",  type=float, default=0.30)
    args = p.parse_args()

    model = None
    if args.ckpt:
        import torch, segmentation_models_pytorch as smp
        ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = smp.Unet(encoder_name=args.encoder, in_channels=3, classes=1)
        model.load_state_dict(ckpt["state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
    else:
        device = "cpu"

    generate_dataset(
        args.img_dir, args.out_dir, args.grade_csv,
        args.gt_he_dir, args.gt_ex_dir, args.gt_ma_dir,
        model, args.threshold, device,
    )
