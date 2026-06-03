#!/usr/bin/env python3
"""
Full-resolution analysis on a trained HE detector checkpoint.
Produces: threshold sweep (with optional TTA), per-image preservation histogram,
worst-5 / best-5 visualisations. Outputs saved locally and optionally to Drive.

Usage:
    python scripts/run_analysis.py \
        --ckpt      /path/to/he_detector_best.pth \
        --idrid_dir "/path/to/A. Segmentation" \
        --out_dir   /path/to/output \
        --tta \
        --drive_dir "gdrive:he_detector_v2/he_run4/analysis/"
"""
import argparse, subprocess
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import segmentation_models_pytorch as smp
from skimage import measure as sk_measure


# ── Inference ─────────────────────────────────────────────────────────────────

def get_prob_map(mdl, img_rgb, patch=768, stride=640, device="cpu"):
    mdl.eval()
    H, W     = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - patch, stride)) + [H - patch]
    xs = list(range(0, W - patch, stride)) + [W - patch]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0   = max(0, min(y0, H - patch))
                x0   = max(0, min(x0, W - patch))
                crop = img_rgb[y0:y0+patch, x0:x0+patch]
                t    = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                t    = t.unsqueeze(0).to(device)
                p    = torch.sigmoid(mdl(t))[0, 0].cpu().numpy()
                prob_map[y0:y0+patch, x0:x0+patch] += p
                count[y0:y0+patch, x0:x0+patch]    += 1
    return prob_map / np.maximum(count, 1)


def get_prob_map_tta(mdl, img_rgb, patch=768, stride=640, device="cpu"):
    """4-way TTA: original + H-flip + V-flip + HV-flip."""
    def run(img):
        return get_prob_map(mdl, img, patch, stride, device)

    p0 = run(img_rgb)
    p1 = run(img_rgb[:, ::-1, :].copy())[:, ::-1]
    p2 = run(img_rgb[::-1, :, :].copy())[::-1, :]
    p3 = run(img_rgb[::-1, ::-1, :].copy())[::-1, ::-1]
    return (p0 + p1 + p2 + p3) / 4.0


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_iou(pred_bin, gt_bin, iou_thresh=0.3):
    gt_labeled   = sk_measure.label(gt_bin)
    pred_labeled = sk_measure.label(pred_bin)
    gt_regions   = sk_measure.regionprops(gt_labeled)
    pred_areas   = {r.label: r.area for r in sk_measure.regionprops(pred_labeled)}
    if not gt_regions:
        return None, 0, 0
    hits = 0
    for gt_reg in gt_regions:
        r0, c0, r1, c1 = gt_reg.bbox
        gt_blob   = (gt_labeled[r0:r1, c0:c1] == gt_reg.label).astype(np.uint8)
        pred_crop = pred_labeled[r0:r1, c0:c1]
        overlap   = np.unique(pred_crop[gt_blob > 0])
        overlap   = overlap[overlap > 0]
        best_iou  = 0.0
        for pl in overlap:
            inter    = (gt_blob & (pred_crop == pl)).sum()
            union    = gt_reg.area + pred_areas.get(pl, 0) - inter
            best_iou = max(best_iou, inter / max(union, 1))
        if best_iou >= iou_thresh:
            hits += 1
    return hits / len(gt_regions), hits, len(gt_regions)


def pixel_preservation(pred_bin, gt_bin, dilate_px=15):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    pred_d = cv2.dilate(pred_bin.astype(np.uint8), kernel)
    total  = int(gt_bin.sum())
    if total == 0:
        return None
    return float((gt_bin & pred_d).sum()) / total


# ── Visualisation ─────────────────────────────────────────────────────────────

def make_viz(stems, imgs, gts, prob_maps, per_image_recall, thr, title, out_path):
    fig, axes = plt.subplots(len(stems), 3, figsize=(15, 5 * len(stems)))
    fig.suptitle(title, fontsize=13)
    for row, stem in enumerate(stems):
        img     = imgs[stem]
        gt      = gts[stem]
        pred    = (prob_maps[stem] > thr).astype(np.uint8) * 255
        overlay = img.copy()
        gt_b = gt > 0; pr_b = pred > 0
        overlay[gt_b & ~pr_b] = [0, 200, 0]
        overlay[pr_b & ~gt_b] = [200, 0, 0]
        overlay[gt_b & pr_b]  = [200, 200, 0]
        r, h, t = per_image_recall.get(stem, (0, 0, 0))
        axes[row, 0].imshow(img);             axes[row, 0].set_title(f"{stem}"); axes[row, 0].axis("off")
        axes[row, 1].imshow(gt, cmap="gray"); axes[row, 1].set_title("GT");     axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"recall={r:.3f} ({h}/{t}) | green=miss  red=FP  yellow=TP")
        axes[row, 2].axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    test_img_dir = Path(args.idrid_dir) / "1. Original Images" / "b. Testing Set"
    test_he_dir  = (Path(args.idrid_dir) / "2. All Segmentation Groundtruths" /
                    "b. Testing Set" / "2. Haemorrhages")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  TTA: {args.tta}")

    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name=args.encoder, in_channels=3, classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch')}  "
          f"recall={ckpt.get('recall', 0):.4f}  "
          f"combined={ckpt.get('combined', 'N/A')}")

    # ── Inference ─────────────────────────────────────────────────────────────
    img_paths = sorted(list(test_img_dir.glob("*.jpg")) +
                       list(test_img_dir.glob("*.jpeg")) +
                       list(test_img_dir.glob("*.png")))
    prob_maps = {}; gts = {}; imgs = {}

    for img_path in img_paths:
        stem    = img_path.stem
        gt_path = next(test_he_dir.glob(f"{stem}_HE.*"), None)
        gt      = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path else None
        img     = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        print(f"  {stem}")
        prob_maps[stem] = (get_prob_map_tta(model, img, args.patch, args.stride, device)
                           if args.tta else
                           get_prob_map(model, img, args.patch, args.stride, device))
        gts[stem]  = gt
        imgs[stem] = img

    # ── Threshold sweep ───────────────────────────────────────────────────────
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"\n{'Threshold':>10}  {'Recall@0.3':>10}  {'N':>4}")
    best_thr = args.threshold; best_recall = 0.0

    for thr in thresholds:
        recalls = []
        for stem, prob in prob_maps.items():
            gt = gts[stem]
            if gt is None or gt.max() == 0:
                continue
            r, _, _ = recall_at_iou((prob > thr).astype(np.uint8), (gt > 0).astype(np.uint8))
            if r is not None:
                recalls.append(r)
        agg = np.mean(recalls) if recalls else 0.0
        print(f"{thr:>10.2f}  {agg:>10.4f}  {len(recalls):>4}")
        if agg > best_recall:
            best_recall = agg; best_thr = thr

    print(f"\nBest threshold: {best_thr}  recall: {best_recall:.4f}")

    # ── Per-image recall at best threshold ────────────────────────────────────
    per_image_recall = {}
    for stem, prob in prob_maps.items():
        gt = gts[stem]
        if gt is None or gt.max() == 0:
            continue
        r, h, t = recall_at_iou((prob > best_thr).astype(np.uint8), (gt > 0).astype(np.uint8))
        if r is not None:
            per_image_recall[stem] = (r, h, t)

    ranked = sorted(per_image_recall.items(), key=lambda x: x[1][0])
    worst5 = [s for s, _ in ranked[:5]]
    best5  = [s for s, _ in ranked[-5:]][::-1]

    print("\nWorst 5:"); [print(f"  {s}: {per_image_recall[s][0]:.3f} ({per_image_recall[s][1]}/{per_image_recall[s][2]})") for s in worst5]
    print("Best 5:");  [print(f"  {s}: {per_image_recall[s][0]:.3f} ({per_image_recall[s][1]}/{per_image_recall[s][2]})") for s in best5]

    # ── Preservation audit ────────────────────────────────────────────────────
    pres_dict = {}
    for stem, prob in prob_maps.items():
        gt = gts[stem]
        if gt is None or gt.max() == 0:
            continue
        p = pixel_preservation((prob > best_thr).astype(np.uint8),
                                (gt > 0).astype(np.uint8), dilate_px=args.dilate_px)
        pres_dict[stem] = p

    values  = [v for v in pres_dict.values() if v is not None]
    mean_p  = np.mean(values)
    below90 = sum(1 for v in values if v < 0.90)
    below85 = [(s, v) for s, v in pres_dict.items() if v is not None and v < 0.85]
    worst_s = min(below85, key=lambda x: x[1]) if below85 else None

    print(f"\n=== Preservation (thr={best_thr}, dilate={args.dilate_px}px, TTA={args.tta}) ===")
    print(f"  mean={mean_p:.4f}  below_0.90={below90}/{len(values)}  "
          f"below_0.85={len(below85)}/{len(values)}  "
          f"worst={worst_s[0]}@{worst_s[1]:.4f}" if worst_s else
          f"  mean={mean_p:.4f}  below_0.90={below90}/{len(values)}  "
          f"below_0.85={len(below85)}/{len(values)}  worst=none")
    print("\nPer-image:")
    for stem, v in sorted(pres_dict.items(), key=lambda x: (x[1] or 0)):
        if v is not None:
            flag = " ← TAIL" if v < 0.85 else (" ← low" if v < 0.90 else "")
            print(f"  {stem}: {v:.4f}{flag}")

    # ── Histogram ─────────────────────────────────────────────────────────────
    worst_v = min(values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
    ax.axvline(0.90, color="orange", linestyle="--", linewidth=1.5, label="0.90")
    ax.axvline(0.85, color="red",    linestyle="--", linewidth=1.5, label="0.85")
    ax.axvline(mean_p, color="green", linestyle="-", linewidth=2, label=f"mean={mean_p:.4f}")
    ax.set_xlabel("Per-image HE GT preservation"); ax.set_ylabel("Image count")
    ax.set_title(f"HE preservation  n={len(values)}  below_0.90={below90}  "
                 f"below_0.85={len(below85)}  worst={worst_v:.4f}")
    ax.legend(); plt.tight_layout()
    hist_path = out / "he_preservation_histogram.png"
    plt.savefig(str(hist_path), dpi=100, bbox_inches="tight"); plt.close()
    print(f"\nSaved: {hist_path}")

    # ── Visualisations ────────────────────────────────────────────────────────
    make_viz(worst5, imgs, gts, prob_maps, per_image_recall, best_thr,
             f"Worst 5  thr={best_thr}  TTA={args.tta}", out / "worst5.png")
    make_viz(best5, imgs, gts, prob_maps, per_image_recall, best_thr,
             f"Best 5   thr={best_thr}  TTA={args.tta}", out / "best5.png")

    # ── Drive upload ──────────────────────────────────────────────────────────
    if args.drive_dir:
        r = subprocess.run(["rclone", "copy", str(out), args.drive_dir],
                           capture_output=True, text=True)
        print("Drive upload:", "OK" if r.returncode == 0 else r.stderr.strip())

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--idrid_dir",  required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--encoder",    default="efficientnet-b2")
    p.add_argument("--patch",      type=int,   default=768)
    p.add_argument("--stride",     type=int,   default=640)
    p.add_argument("--threshold",  type=float, default=0.05)
    p.add_argument("--dilate_px",  type=int,   default=15)
    p.add_argument("--tta",        action="store_true")
    p.add_argument("--drive_dir",  default=None)
    main(p.parse_args())
