"""
check_od_contamination.py

Checks how much of each learned EX checkpoint's false positives fall on the
optic disc, across the full IDRiD test set, and how much precision improves
if the same optic-disc exclusion mask used by the rule-based detector is
applied post-hoc to the learned detector's predictions.

Usage:
    python check_od_contamination.py --ckpt /workspace/models/ex_detector/ex_detector_best.pth --label seed42
"""
import argparse
import sys
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from skimage import measure as sk_measure

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--label", required=True)
args = p.parse_args()

sys.path.insert(0, "/workspace/realizemd_deid")
from pipeline.pathology import detect_optic_disc

BASE = Path("/workspace/data/idrid/A. Segmentation")
IMAGES = BASE / "1. Original Images" / "b. Testing Set"
GT_ROOT = BASE / "2. All Segmentation Groundtruths" / "b. Testing Set" / "3. Hard Exudates"

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()
print(f"[{args.label}] Loaded: epoch={ckpt.get('epoch')} seed={ckpt.get('seed')}")


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def get_prob_map(img_rgb, patch=512, stride=384):
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
                prob = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0c:y0c+patch, x0c:x0c+patch] += prob
                count[y0c:y0c+patch, x0c:x0c+patch] += 1
    return prob_map / np.maximum(count, 1)


total_fp = 0
total_fp_on_od = 0
total_tp = 0
total_pred = 0
total_tp_masked = 0
total_pred_masked = 0
images_with_od_fp = []

for img_path in sorted(IMAGES.glob("*.jpg")):
    stem = img_path.stem
    gt_path = next(GT_ROOT.glob(f"{stem}_EX.*"), None)
    if gt_path is None:
        continue
    img_bgr = cv2.imread(str(img_path))
    img_rgb = preprocess(img_bgr)
    od_mask = detect_optic_disc(img_rgb)  # 255 = optic disc region

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt > 0).astype(np.uint8)

    prob = get_prob_map(img_rgb)
    pred_bin = (prob > 0.35).astype(np.uint8)

    tp = int((gt_bin & pred_bin).sum())
    pred_px = int(pred_bin.sum())
    fp_mask = (pred_bin > 0) & (gt_bin == 0)
    fp = int(fp_mask.sum())
    fp_on_od = int((fp_mask & (od_mask > 0)).sum())

    # precision with OD-excluded predictions (same mask rule-based EX uses)
    pred_masked = cv2.bitwise_and(pred_bin, cv2.bitwise_not(od_mask))
    tp_masked = int((gt_bin & (pred_masked > 0)).sum())
    pred_masked_px = int((pred_masked > 0).sum())

    total_fp += fp
    total_fp_on_od += fp_on_od
    total_tp += tp
    total_pred += pred_px
    total_tp_masked += tp_masked
    total_pred_masked += pred_masked_px

    if fp_on_od > 500:
        images_with_od_fp.append((stem, fp_on_od, fp))

    print(f"{stem:16s} pred_px={pred_px:>7} fp={fp:>7} fp_on_od={fp_on_od:>7} "
          f"({100*fp_on_od/max(fp,1):.1f}% of FP)")

print(f"\n=== [{args.label}] Summary ===")
print(f"Total FP pixels:            {total_fp}")
print(f"Total FP on optic disc:     {total_fp_on_od}  ({100*total_fp_on_od/max(total_fp,1):.1f}% of all FP)")
print(f"Precision (raw):            {total_tp/max(total_pred,1):.4f}")
print(f"Precision (OD-excluded):    {total_tp_masked/max(total_pred_masked,1):.4f}")
print(f"Images with significant OD contamination (>500px): {len(images_with_od_fp)}")
for stem, fp_od, fp in images_with_od_fp:
    print(f"  {stem}: {fp_od}px on OD out of {fp}px total FP")
