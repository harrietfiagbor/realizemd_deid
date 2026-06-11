"""
Lesion preservation evaluator for the de-identification pipeline.

After the synthesis model de-identifies an image (scrambles identity, keeps
disease subspace), this evaluator checks that the lesions are still present
in the regenerated image — correct type, location, and morphology.

This is the clinical-utility gate for Ichilov:
  "After de-identification, does the image still show the same pathology?"

Metrics per image:
  HE lesions (via detector):
  - per-type recall  : fraction of GT lesion blobs detected in de-id image
  - centroid shift   : mean pixel distance between matched GT and de-id blobs
  - area ratio       : median(de-id blob area / GT blob area) — morphology check
  - pixel overlap    : GT mask pixels covered by dilated de-id prediction

  Optic disc (via GT mask region, no detector needed):
  - disc_intensity_delta : mean absolute intensity difference inside OD region
                           between original and de-id image. Measures how much
                           the disc appearance changed — critical because the disc
                           is both identity-dense (Azizi) and clinically load-bearing
                           for glaucoma. Should stay near 0 after de-id.
  - disc_ssim            : structural similarity inside the OD bounding box.
                           1.0 = identical structure, lower = morphology changed.

Uses the trained HE detector checkpoint (run_analysis.py inference logic)
for HE detection. EX/MA can be added when those detectors are trained.

Usage:
    python lesion_preservation_eval.py \
        --orig_dir   /path/to/original/images \
        --deid_dir   /path/to/deid/images \
        --gt_dir     "/path/to/A. Segmentation/.../2. Haemorrhages" \
        --od_dir     "/path/to/A. Segmentation/.../5. Optic Disc" \
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


# ── Preprocessing (must match training) ──────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


# ── Inference (sliding window, matches run_analysis.py) ──────────────────────

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


# ── Per-blob matching ─────────────────────────────────────────────────────────

def match_blobs(gt_bin, pred_bin, iou_thresh=0.3):
    """
    Match GT blobs to predicted blobs by IoU.

    Returns:
        hits         : number of GT blobs matched
        n_gt         : total GT blobs
        centroid_shifts : list of pixel distances for matched pairs
        area_ratios     : list of (pred_area / gt_area) for matched pairs
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
            # centroid shift in pixels
            gy, gx = gt_r.centroid
            if best_pl in pred_cents:
                py, px = pred_cents[best_pl]
                centroid_shifts.append(np.sqrt((gy - py) ** 2 + (gx - px) ** 2))
            # area ratio
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


# ── Optic disc readout ────────────────────────────────────────────────────────

def disc_readout(orig_rgb, deid_rgb, od_mask):
    """
    Compare the optic disc region between original and de-id image.

    The disc is clinically critical for glaucoma AND identity-dense (Azizi).
    We don't run a detector — we use the GT OD mask as the region of interest
    and measure how much the disc appearance changed after de-identification.

    Args:
        orig_rgb : (H, W, 3) float32 preprocessed original image
        deid_rgb : (H, W, 3) float32 preprocessed de-id image
        od_mask  : (H, W)    binary uint8 GT optic disc mask

    Returns dict:
        disc_intensity_delta : mean absolute pixel difference inside OD region
                               (target: as low as possible — disc should look the same)
        disc_ssim            : structural similarity inside OD bounding box
                               (target: close to 1.0 — disc structure preserved)
        disc_area_px         : OD mask area in pixels (context)
    """
    if od_mask is None or od_mask.max() == 0:
        return dict(disc_intensity_delta=None, disc_ssim=None, disc_area_px=0)

    # Bounding box of OD region
    rows = np.any(od_mask, axis=1); cols = np.any(od_mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    orig_crop = orig_rgb[r0:r1+1, c0:c1+1]
    deid_crop = deid_rgb[r0:r1+1, c0:c1+1]
    mask_crop = od_mask[r0:r1+1, c0:c1+1].astype(bool)

    # Mean absolute intensity difference inside OD pixels only
    diff = np.abs(orig_crop.astype(np.float32) - deid_crop.astype(np.float32))
    intensity_delta = float(diff[mask_crop].mean()) if mask_crop.any() else None

    # SSIM on the bounding box crop (grayscale)
    orig_gray = cv2.cvtColor((orig_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    deid_gray = cv2.cvtColor((deid_crop * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    win = min(7, orig_gray.shape[0] - 1, orig_gray.shape[1] - 1)
    win = win if win % 2 == 1 else win - 1  # must be odd
    disc_ssim_val = float(ssim(orig_gray, deid_gray, win_size=max(win, 3),
                               data_range=255)) if win >= 3 else None

    return dict(
        disc_intensity_delta = intensity_delta,
        disc_ssim            = disc_ssim_val,
        disc_area_px         = int(od_mask.sum()),
    )


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(orig_dir, deid_dir, gt_dir, od_dir, model, thr, dilate_px, device):
    """
    Run preservation eval on all image pairs in orig_dir / deid_dir.

    Returns list of per-image result dicts.
    """
    results = []
    orig_paths = sorted(Path(orig_dir).glob("*.jpg")) + \
                 sorted(Path(orig_dir).glob("*.png"))

    for orig_path in orig_paths:
        stem = orig_path.stem
        deid_path = next(
            (Path(deid_dir) / f"{stem}{ext}" for ext in (".jpg", ".png")
             if (Path(deid_dir) / f"{stem}{ext}").exists()), None)
        gt_path = next(
            (Path(gt_dir) / f"{stem}_HE{ext}" for ext in (".tif", ".png")
             if (Path(gt_dir) / f"{stem}_HE{ext}").exists()), None)

        if deid_path is None or gt_path is None:
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None or gt.max() == 0:
            continue  # no lesions in this image

        orig_img = preprocess(cv2.imread(str(orig_path)))
        deid_img = preprocess(cv2.imread(str(deid_path)))

        prob_orig = get_prob_map(model, orig_img, device=device)
        prob_deid = get_prob_map(model, deid_img, device=device)

        gt_bin    = (gt > 0).astype(np.uint8)
        pred_orig = (prob_orig > thr).astype(np.uint8)
        pred_deid = (prob_deid > thr).astype(np.uint8)

        hits_orig, n_gt, _, _          = match_blobs(gt_bin, pred_orig)
        hits_deid, _,    shifts, areas = match_blobs(gt_bin, pred_deid)

        pres = pixel_preservation(pred_deid, gt_bin, dilate_px)

        # Optic disc readout
        od_mask = None
        if od_dir:
            od_path = next(
                (Path(od_dir) / f"{stem}_OD{ext}" for ext in (".tif", ".png")
                 if (Path(od_dir) / f"{stem}_OD{ext}").exists()), None)
            if od_path:
                od_raw = cv2.imread(str(od_path), cv2.IMREAD_GRAYSCALE)
                od_mask = (od_raw > 0).astype(np.uint8) if od_raw is not None else None
        disc = disc_readout(orig_img, deid_img, od_mask)

        results.append(dict(
            patient_id           = stem,
            n_gt_blobs           = n_gt,
            recall_orig          = hits_orig / max(n_gt, 1),
            recall_deid          = hits_deid / max(n_gt, 1),
            recall_drop          = (hits_orig - hits_deid) / max(n_gt, 1),
            mean_centroid_shift  = float(np.mean(shifts)) if shifts else None,
            median_area_ratio    = float(np.median(areas)) if areas else None,
            pixel_preservation   = pres,
            disc_intensity_delta = disc["disc_intensity_delta"],
            disc_ssim            = disc["disc_ssim"],
            disc_area_px         = disc["disc_area_px"],
        ))
        d_ssim = f"{disc['disc_ssim']:.3f}" if disc["disc_ssim"] is not None else "N/A"
        print(f"  {stem}: he_recall={results[-1]['recall_deid']:.3f}  "
              f"pres={pres:.4f}  disc_ssim={d_ssim}" if pres else
              f"  {stem}: he_recall={results[-1]['recall_deid']:.3f}  "
              f"pres=N/A  disc_ssim={d_ssim}")

    return results


def print_summary(results):
    recall_orig = [r["recall_orig"] for r in results]
    recall_deid = [r["recall_deid"] for r in results]
    drops       = [r["recall_drop"] for r in results]
    pres        = [r["pixel_preservation"] for r in results if r["pixel_preservation"] is not None]
    shifts      = [r["mean_centroid_shift"] for r in results if r["mean_centroid_shift"] is not None]
    areas       = [r["median_area_ratio"] for r in results if r["median_area_ratio"] is not None]
    disc_deltas = [r["disc_intensity_delta"] for r in results if r["disc_intensity_delta"] is not None]
    disc_ssims  = [r["disc_ssim"] for r in results if r["disc_ssim"] is not None]

    print(f"\n=== Lesion Preservation Summary ({len(results)} images) ===")
    print(f"  HE recall on original  : {np.mean(recall_orig):.4f}")
    print(f"  HE recall on de-id     : {np.mean(recall_deid):.4f}")
    print(f"  Mean recall drop       : {np.mean(drops):.4f}  (target: < 0.05)")
    print(f"  Mean pixel pres.       : {np.mean(pres):.4f}   (target: >= 0.98)")
    print(f"  Mean centroid shift    : {np.mean(shifts):.1f}px" if shifts else "  Centroid shift: N/A")
    print(f"  Median area ratio      : {np.median(areas):.3f}  (1.0 = perfect morphology)" if areas else "  Area ratio: N/A")

    print(f"\n  === Optic Disc Readout ===")
    if disc_ssims:
        print(f"  Mean disc SSIM         : {np.mean(disc_ssims):.4f}  (1.0 = structure preserved)")
        print(f"  Mean disc intensity d  : {np.mean(disc_deltas):.4f}  (0.0 = appearance unchanged)")
        disc_tail = [(r["patient_id"], r["disc_ssim"]) for r in results
                     if r["disc_ssim"] is not None and r["disc_ssim"] < 0.80]
        if disc_tail:
            print(f"  Images with disc SSIM < 0.80 ({len(disc_tail)}) — identity may be leaking:")
            for pid, s in sorted(disc_tail, key=lambda x: x[1]):
                print(f"    {pid}: disc_ssim={s:.3f}")
    else:
        print("  No OD masks provided — skipped disc readout")

    # Flag images where de-id recall drops > 10%
    tail = [(r["patient_id"], r["recall_drop"]) for r in results if r["recall_drop"] > 0.10]
    if tail:
        print(f"\n  Images with >10% HE recall drop ({len(tail)}):")
        for pid, drop in sorted(tail, key=lambda x: -x[1]):
            print(f"    {pid}: drop={drop:.3f}")


def save_histogram(results, out_path):
    pres = [r["pixel_preservation"] for r in results if r["pixel_preservation"] is not None]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pres, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
    ax.axvline(0.98, color="red",    linestyle="--", linewidth=1.5, label="target 0.98")
    ax.axvline(np.mean(pres), color="green", linestyle="-", linewidth=2,
               label=f"mean={np.mean(pres):.4f}")
    ax.set_xlabel("Pixel preservation (de-id vs GT)")
    ax.set_ylabel("Image count")
    ax.set_title("Lesion preservation after de-identification")
    ax.legend(); plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight"); plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name=args.encoder, in_channels=3, classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch')}  recall={ckpt.get('recall', 0):.4f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    results = evaluate(args.orig_dir, args.deid_dir, args.gt_dir, args.od_dir,
                       model, args.threshold, args.dilate_px, device)

    print_summary(results)
    save_histogram(results, out / "preservation_histogram.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--orig_dir",   required=True, help="Folder of original images")
    p.add_argument("--deid_dir",   required=True, help="Folder of de-identified images")
    p.add_argument("--gt_dir",     required=True, help="Folder of GT HE mask .tif files")
    p.add_argument("--od_dir",     default=None,  help="Folder of GT optic disc mask .tif files (optional)")
    p.add_argument("--ckpt",       required=True, help="HE detector checkpoint .pth")
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--encoder",    default="efficientnet-b2")
    p.add_argument("--threshold",  type=float, default=0.30)
    p.add_argument("--dilate_px",  type=int,   default=15)
    main(p.parse_args())
