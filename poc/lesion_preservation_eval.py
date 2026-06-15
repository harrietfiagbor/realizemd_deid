"""
Lesion preservation evaluator for the de-identification pipeline.

After the synthesis model de-identifies an image (scrambles identity, keeps
disease subspace), this evaluator checks that the lesions are still present
in the regenerated image — correct type, location, and morphology.

This is the clinical-utility gate for Ichilov:
  "After de-identification, does the image still show the same pathology?"

Metrics per image:
  HE lesions (via trained detector):
  - per-type recall  : fraction of GT lesion blobs detected in de-id image
  - centroid shift   : mean pixel distance between matched GT and de-id blobs
  - area ratio       : median(de-id blob area / GT blob area) — morphology check
  - pixel overlap    : GT mask pixels covered by dilated de-id prediction

  EX / MA lesions (via rule-based detectors — no checkpoint needed):
  - same metrics as HE above, using GT masks from IDRiD

  Optic disc:
  - disc_intensity_delta : mean absolute intensity difference inside OD region
  - disc_ssim            : structural similarity inside the OD bounding box

Usage:
    python lesion_preservation_eval.py \
        --orig_dir   /path/to/original/images \
        --deid_dir   /path/to/deid/images \
        --gt_he_dir  "/path/to/.../2. Haemorrhages" \
        --gt_ex_dir  "/path/to/.../3. Hard Exudates" \
        --gt_ma_dir  "/path/to/.../1. Microaneurysms" \
        --od_dir     "/path/to/.../5. Optic Disc" \
        --ckpt       /path/to/he_detector_best.pth \
        --out_dir    /path/to/output \
        --dilate_px  15
"""
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from skimage import measure as sk_measure
from skimage.metrics import structural_similarity as ssim

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.pathology import detect_hard_exudates, detect_microaneurysms, detect_optic_disc


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


# ── HE inference (sliding window) ────────────────────────────────────────────

def get_prob_map(model, img_rgb, patch=768, stride=640, device="cpu"):
    model.eval()
    H, W     = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - patch, stride)) + [H - patch]
    xs = list(range(0, W - patch, stride)) + [W - patch]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0 = max(0, min(y0, H - patch))
                x0 = max(0, min(x0, W - patch))
                crop = img_rgb[y0:y0+patch, x0:x0+patch]
                t    = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                p    = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0:y0+patch, x0:x0+patch] += p
                count[y0:y0+patch, x0:x0+patch]    += 1
    return prob_map / np.maximum(count, 1)


# ── Shared metrics ────────────────────────────────────────────────────────────

def match_blobs(gt_bin, pred_bin, iou_thresh=0.3):
    """
    Match GT blobs to predicted blobs by IoU.
    Returns: hits, n_gt, centroid_shifts, area_ratios
    """
    gt_lab   = sk_measure.label(gt_bin)
    pred_lab = sk_measure.label(pred_bin)
    gt_regs  = sk_measure.regionprops(gt_lab)
    pred_areas = {r.label: r.area for r in sk_measure.regionprops(pred_lab)}
    pred_cents = {r.label: r.centroid for r in sk_measure.regionprops(pred_lab)}

    hits, centroid_shifts, area_ratios = 0, [], []
    for gt_r in gt_regs:
        r0, c0, r1, c1 = gt_r.bbox
        gt_blob   = (gt_lab[r0:r1, c0:c1] == gt_r.label).astype(np.uint8)
        pred_crop = pred_lab[r0:r1, c0:c1]
        overlap   = np.unique(pred_crop[gt_blob > 0])
        overlap   = overlap[overlap > 0]
        best_iou, best_pl = 0.0, None
        for pl in overlap:
            inter = (gt_blob & (pred_crop == pl)).sum()
            union = gt_r.area + pred_areas.get(pl, 0) - inter
            iou   = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_pl = iou, pl
        if best_iou >= iou_thresh:
            hits += 1
            gy, gx = gt_r.centroid
            if best_pl in pred_cents:
                py, px = pred_cents[best_pl]
                centroid_shifts.append(np.sqrt((gy - py) ** 2 + (gx - px) ** 2))
            if best_pl in pred_areas:
                area_ratios.append(pred_areas[best_pl] / max(gt_r.area, 1))

    return hits, len(gt_regs), centroid_shifts, area_ratios


def pixel_preservation(pred_bin, gt_bin, dilate_px=15):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    pred_d = cv2.dilate(pred_bin.astype(np.uint8), kernel)
    total  = int(gt_bin.sum())
    if total == 0:
        return None
    return float((gt_bin & pred_d).sum()) / total


def _lesion_metrics(gt_bin, pred_orig_bin, pred_deid_bin, dilate_px):
    """Compute full metric set for one lesion type."""
    hits_orig, n_gt, _, _          = match_blobs(gt_bin, pred_orig_bin)
    hits_deid, _,    shifts, areas = match_blobs(gt_bin, pred_deid_bin)
    pres = pixel_preservation(pred_deid_bin, gt_bin, dilate_px)
    return dict(
        n_gt          = n_gt,
        recall_orig   = hits_orig / max(n_gt, 1),
        recall_deid   = hits_deid / max(n_gt, 1),
        recall_drop   = (hits_orig - hits_deid) / max(n_gt, 1),
        mean_shift    = float(np.mean(shifts)) if shifts else None,
        median_area_r = float(np.median(areas)) if areas else None,
        pixel_pres    = pres,
    )


# ── Optic disc readout ────────────────────────────────────────────────────────

def disc_readout(orig_rgb, deid_rgb, od_mask):
    if od_mask is None or od_mask.max() == 0:
        return dict(disc_intensity_delta=None, disc_ssim=None, disc_area_px=0)

    rows = np.any(od_mask, axis=1); cols = np.any(od_mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    orig_crop = orig_rgb[r0:r1+1, c0:c1+1]
    deid_crop = deid_rgb[r0:r1+1, c0:c1+1]
    mask_crop = od_mask[r0:r1+1, c0:c1+1].astype(bool)

    diff = np.abs(orig_crop.astype(np.float32) - deid_crop.astype(np.float32))
    intensity_delta = float(diff[mask_crop].mean()) if mask_crop.any() else None

    orig_gray = cv2.cvtColor((orig_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    deid_gray = cv2.cvtColor((deid_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    win = min(7, orig_gray.shape[0] - 1, orig_gray.shape[1] - 1)
    win = win if win % 2 == 1 else win - 1
    disc_ssim_val = float(ssim(orig_gray, deid_gray, win_size=max(win, 3),
                               data_range=255)) if win >= 3 else None

    return dict(
        disc_intensity_delta = intensity_delta,
        disc_ssim            = disc_ssim_val,
        disc_area_px         = int(od_mask.sum()),
    )


# ── GT mask loader ────────────────────────────────────────────────────────────

def _load_gt(gt_dir, stem, suffix):
    """Load GT mask; returns binary uint8 array or None."""
    if gt_dir is None:
        return None
    for ext in (".tif", ".png"):
        p = Path(gt_dir) / f"{stem}{suffix}{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.max() > 0:
                return (m > 0).astype(np.uint8)
    return None


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(orig_dir, deid_dir, gt_he_dir, gt_ex_dir, gt_ma_dir,
             od_dir, model, thr, dilate_px, device):
    results = []
    orig_paths = sorted(Path(orig_dir).glob("*.jpg")) + \
                 sorted(Path(orig_dir).glob("*.png"))

    for orig_path in orig_paths:
        stem = orig_path.stem
        deid_path = next(
            (Path(deid_dir) / f"{stem}{ext}" for ext in (".jpg", ".png")
             if (Path(deid_dir) / f"{stem}{ext}").exists()), None)
        if deid_path is None:
            continue

        orig_img = preprocess(cv2.imread(str(orig_path)))
        deid_img = preprocess(cv2.imread(str(deid_path)))

        # ── HE (learned detector) ──────────────────────────────────────────
        gt_he = _load_gt(gt_he_dir, stem, "_HE")
        he = dict(n_gt=0, recall_orig=None, recall_deid=None,
                  recall_drop=None, mean_shift=None, median_area_r=None, pixel_pres=None)
        if gt_he is not None and model is not None:
            prob_orig = get_prob_map(model, orig_img, device=device)
            prob_deid = get_prob_map(model, deid_img, device=device)
            he = _lesion_metrics(gt_he,
                                 (prob_orig > thr).astype(np.uint8),
                                 (prob_deid > thr).astype(np.uint8),
                                 dilate_px)

        # ── EX + MA (rule-based) ───────────────────────────────────────────
        od_orig = detect_optic_disc(orig_img)
        od_deid = detect_optic_disc(deid_img)

        pred_ex_orig = detect_hard_exudates(orig_img, od_orig)
        pred_ex_deid = detect_hard_exudates(deid_img, od_deid)
        pred_ma_orig = detect_microaneurysms(orig_img, od_orig)
        pred_ma_deid = detect_microaneurysms(deid_img, od_deid)

        gt_ex = _load_gt(gt_ex_dir, stem, "_EX")
        gt_ma = _load_gt(gt_ma_dir, stem, "_MA")

        ex = _lesion_metrics(gt_ex,
                             (pred_ex_orig > 0).astype(np.uint8),
                             (pred_ex_deid > 0).astype(np.uint8),
                             dilate_px) if gt_ex is not None else None

        ma = _lesion_metrics(gt_ma,
                             (pred_ma_orig > 0).astype(np.uint8),
                             (pred_ma_deid > 0).astype(np.uint8),
                             dilate_px) if gt_ma is not None else None

        # ── Optic disc readout ─────────────────────────────────────────────
        od_mask_gt = None
        if od_dir:
            od_path = next(
                (Path(od_dir) / f"{stem}_OD{ext}" for ext in (".tif", ".png")
                 if (Path(od_dir) / f"{stem}_OD{ext}").exists()), None)
            if od_path:
                raw = cv2.imread(str(od_path), cv2.IMREAD_GRAYSCALE)
                od_mask_gt = (raw > 0).astype(np.uint8) if raw is not None else None
        disc = disc_readout(orig_img, deid_img, od_mask_gt)

        results.append(dict(
            patient_id = stem,
            he=he, ex=ex, ma=ma,
            disc_intensity_delta = disc["disc_intensity_delta"],
            disc_ssim            = disc["disc_ssim"],
            disc_area_px         = disc["disc_area_px"],
        ))

        he_str = f"he={he['recall_deid']:.3f}" if he['recall_deid'] is not None else "he=N/A"
        ex_str = f"ex={ex['recall_deid']:.3f}" if ex is not None else "ex=N/A"
        ma_str = f"ma={ma['recall_deid']:.3f}" if ma is not None else "ma=N/A"
        d_ssim = f"disc_ssim={disc['disc_ssim']:.3f}" if disc["disc_ssim"] else ""
        print(f"  {stem}  {he_str}  {ex_str}  {ma_str}  {d_ssim}")

    return results


def _summarise_type(results, key, label):
    rows = [r[key] for r in results if r[key] is not None]
    if not rows:
        print(f"  {label}: no GT available")
        return
    recall_deid = [r["recall_deid"] for r in rows]
    drops       = [r["recall_drop"] for r in rows]
    pres        = [r["pixel_pres"] for r in rows if r["pixel_pres"] is not None]
    shifts      = [r["mean_shift"] for r in rows if r["mean_shift"] is not None]

    print(f"\n  [{label}]")
    print(f"    Mean recall (de-id)  : {np.mean(recall_deid):.4f}")
    print(f"    Mean recall drop     : {np.mean(drops):.4f}   (target: < 0.05)")
    print(f"    Mean pixel pres.     : {np.mean(pres):.4f}   (target: >= 0.98)" if pres else "    Pixel pres: N/A")
    print(f"    Mean centroid shift  : {np.mean(shifts):.1f}px" if shifts else "    Centroid shift: N/A")

    tail = [(results[i]["patient_id"], r["recall_drop"])
            for i, r in enumerate(rows) if r["recall_drop"] > 0.10]
    if tail:
        print(f"    Images with >10% recall drop ({len(tail)}):")
        for pid, d in sorted(tail, key=lambda x: -x[1]):
            print(f"      {pid}: drop={d:.3f}")


def print_summary(results):
    print(f"\n=== Lesion Preservation Summary ({len(results)} images) ===")
    _summarise_type(results, "he", "HE — haemorrhages (learned detector)")
    _summarise_type(results, "ex", "EX — hard exudates (rule-based)")
    _summarise_type(results, "ma", "MA — microaneurysms (rule-based)")

    disc_ssims  = [r["disc_ssim"]            for r in results if r["disc_ssim"] is not None]
    disc_deltas = [r["disc_intensity_delta"]  for r in results if r["disc_intensity_delta"] is not None]
    print(f"\n  [Optic disc]")
    if disc_ssims:
        print(f"    Mean disc SSIM        : {np.mean(disc_ssims):.4f}  (1.0 = structure preserved)")
        print(f"    Mean disc intensity d : {np.mean(disc_deltas):.4f}  (0.0 = appearance unchanged)")
        tail = [(r["patient_id"], r["disc_ssim"]) for r in results
                if r["disc_ssim"] is not None and r["disc_ssim"] < 0.80]
        if tail:
            print(f"    disc SSIM < 0.80 ({len(tail)}) — potential identity leakage:")
            for pid, s in sorted(tail, key=lambda x: x[1]):
                print(f"      {pid}: disc_ssim={s:.3f}")
    else:
        print("    No OD GT masks provided — skipped disc readout")


def save_histogram(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label, color in zip(
        axes,
        ["he", "ex", "ma"],
        ["HE recall (de-id)", "EX recall (de-id)", "MA recall (de-id)"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        vals = [r[key]["recall_deid"] for r in results
                if r[key] is not None and r[key]["recall_deid"] is not None]
        if not vals:
            ax.set_title(f"{label}\n(no data)")
            continue
        ax.hist(vals, bins=20, range=(0, 1), color=color, edgecolor="black")
        ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_xlabel(label); ax.set_ylabel("Image count")
        ax.set_title(label); ax.legend()
    plt.suptitle("Lesion preservation after de-identification", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    if args.ckpt:
        print(f"Device: {device}")
        ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = smp.Unet(encoder_name=args.encoder, in_channels=3, classes=1)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        print(f"HE checkpoint: epoch={ckpt.get('epoch')}  recall={ckpt.get('recall', 0):.4f}")
    else:
        print("No HE checkpoint provided — HE metrics will be skipped")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    results = evaluate(
        args.orig_dir, args.deid_dir,
        args.gt_he_dir, args.gt_ex_dir, args.gt_ma_dir,
        args.od_dir, model, args.threshold, args.dilate_px, device,
    )

    print_summary(results)
    save_histogram(results, out / "preservation_histogram.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--orig_dir",   required=True)
    p.add_argument("--deid_dir",   required=True)
    p.add_argument("--gt_he_dir",  default=None, help="GT haemorrhage .tif folder")
    p.add_argument("--gt_ex_dir",  default=None, help="GT hard exudate .tif folder")
    p.add_argument("--gt_ma_dir",  default=None, help="GT microaneurysm .tif folder")
    p.add_argument("--od_dir",     default=None, help="GT optic disc .tif folder")
    p.add_argument("--ckpt",       default=None, help="HE detector checkpoint (optional)")
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--encoder",    default="efficientnet-b2")
    p.add_argument("--threshold",  type=float, default=0.30)
    p.add_argument("--dilate_px",  type=int,   default=15)
    main(p.parse_args())
