#!/usr/bin/env python3
"""
check_od_contamination_learned.py

Re-runs the original check_od_contamination.py (victoria-ex-seed-check branch)
methodology, but replaces the brightness-heuristic detect_optic_disc() exclusion
mask with the trained learned OD detector (dice=0.9629 IDRiD, resolution-robust,
EyePACS wrong-blob gap closed -- see decision.md "OD Detector b0..." entries).

Original finding this re-checks: same EX training recipe produced 1.1% / 2.3% /
14.6% optic-disc contamination across seeds 42/123/7 -- a >10x spread Adam called
"random corruption" and the reason universal exclusion was paused pending a
validated localizer. This script measures whether a validated learned OD mask
actually collapses that contamination toward 0 on all three, and reports both
aggregate AND macro-average precision before/after (macro-average is the
project's established working figure for EX precision, not aggregate -- see
decision.md "Resolution Decision Locked").

Usage (per EX checkpoint, run 3x):
    python check_od_contamination_learned.py \
        --ex_ckpt /workspace/models/ex_detector_seed7/ex_detector_best.pth \
        --od_ckpt /workspace/od_detector/models/od_run1_b0/od_detector_best.pth \
        --label seed7
"""
import argparse
import sys
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--ex_ckpt", required=True)
p.add_argument("--od_ckpt", required=True)
p.add_argument("--label", required=True)
p.add_argument("--idrid_dir", default="/workspace/data/idrid/A. Segmentation")
args = p.parse_args()

BASE = Path(args.idrid_dir)
IMAGES = BASE / "1. Original Images" / "b. Testing Set"
GT_ROOT = BASE / "2. All Segmentation Groundtruths" / "b. Testing Set" / "3. Hard Exudates"

device = "cuda" if torch.cuda.is_available() else "cpu"

ex_ckpt = torch.load(args.ex_ckpt, map_location="cpu", weights_only=False)
ex_model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1)
ex_model.load_state_dict(ex_ckpt["state_dict"])
ex_model.to(device).eval()
print(f"[{args.label}] EX loaded: epoch={ex_ckpt.get('epoch')} seed={ex_ckpt.get('seed')}")

od_ckpt = torch.load(args.od_ckpt, map_location="cpu", weights_only=False)
od_model = smp.Unet(encoder_name=od_ckpt["encoder"], in_channels=3, classes=1)
od_model.load_state_dict(od_ckpt["state_dict"])
od_model.to(device).eval()
od_img_size = od_ckpt["img_size"]
print(f"OD loaded: epoch={od_ckpt.get('epoch')} encoder={od_ckpt.get('encoder')} dice={od_ckpt.get('dice'):.4f}")


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def get_ex_prob_map(img_rgb, patch=512, stride=384):
    H, W = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - patch, stride)) + [H - patch]
    xs = list(range(0, W - patch, stride)) + [W - patch]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0c = max(0, min(y0, H - patch))
                x0c = max(0, min(x0, W - patch))
                crop = img_rgb[y0c:y0c+patch, x0c:x0c+patch]
                t = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                prob = torch.sigmoid(ex_model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0c:y0c+patch, x0c:x0c+patch] += prob
                count[y0c:y0c+patch, x0c:x0c+patch] += 1
    return prob_map / np.maximum(count, 1)


def get_od_mask(img_rgb):
    """Whole-image resize inference, matching how the OD model was trained."""
    H, W = img_rgb.shape[:2]
    img_r = cv2.resize(img_rgb, (od_img_size, od_img_size))
    with torch.no_grad():
        t = torch.from_numpy(img_r.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        prob = torch.sigmoid(od_model(t.to(device)))[0, 0].cpu().numpy()
    pred_small = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_NEAREST)


total_fp = total_fp_on_od = total_tp = total_pred = 0
total_tp_masked = total_pred_masked = 0
per_image_precision_raw, per_image_precision_masked = [], []
images_with_od_fp = []

for img_path in sorted(IMAGES.glob("*.jpg")):
    stem = img_path.stem
    gt_path = next(GT_ROOT.glob(f"{stem}_EX.*"), None)
    if gt_path is None:
        continue
    img_bgr = cv2.imread(str(img_path))
    img_rgb = preprocess(img_bgr)
    od_mask = get_od_mask(img_rgb)  # 255 = learned optic disc region

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt > 0).astype(np.uint8)

    prob = get_ex_prob_map(img_rgb)
    pred_bin = (prob > 0.35).astype(np.uint8)

    tp = int((gt_bin & pred_bin).sum())
    pred_px = int(pred_bin.sum())
    fp_mask = (pred_bin > 0) & (gt_bin == 0)
    fp = int(fp_mask.sum())
    fp_on_od = int((fp_mask & (od_mask > 0)).sum())

    pred_masked = cv2.bitwise_and((pred_bin * 255).astype(np.uint8), cv2.bitwise_not(od_mask))
    tp_masked = int((gt_bin & (pred_masked > 0)).sum())
    pred_masked_px = int((pred_masked > 0).sum())

    total_fp += fp
    total_fp_on_od += fp_on_od
    total_tp += tp
    total_pred += pred_px
    total_tp_masked += tp_masked
    total_pred_masked += pred_masked_px

    per_image_precision_raw.append(tp / max(pred_px, 1))
    per_image_precision_masked.append(tp_masked / max(pred_masked_px, 1))

    if fp_on_od > 500:
        images_with_od_fp.append((stem, fp_on_od, fp))

    print(f"{stem:16s} pred_px={pred_px:>7} fp={fp:>7} fp_on_od={fp_on_od:>7} "
          f"({100*fp_on_od/max(fp,1):.1f}% of FP)")

print(f"\n=== [{args.label}] Summary (learned OD exclusion) ===")
print(f"Total FP pixels:                    {total_fp}")
print(f"Total FP on optic disc:             {total_fp_on_od}  ({100*total_fp_on_od/max(total_fp,1):.1f}% of all FP)")
print(f"Precision aggregate (raw):          {total_tp/max(total_pred,1):.4f}")
print(f"Precision aggregate (OD-excluded):  {total_tp_masked/max(total_pred_masked,1):.4f}")
print(f"Precision macro-avg (raw):          {np.mean(per_image_precision_raw):.4f}")
print(f"Precision macro-avg (OD-excluded):  {np.mean(per_image_precision_masked):.4f}")
print(f"Images with significant OD contamination (>500px): {len(images_with_od_fp)}")
for stem, fp_od, fp in images_with_od_fp:
    print(f"  {stem}: {fp_od}px on OD out of {fp}px total FP")
