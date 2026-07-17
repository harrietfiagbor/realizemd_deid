"""
diagnose_ex_tiling.py

Investigates whether a suspicious blocky/rectangular prediction blob is a
sliding-window tiling artifact (patch boundaries visible in the probability
map) or a genuine model failure. Overlays the patch/stride grid on the raw
probability map so misalignment vs artifact-alignment is visible directly.

Usage:
    python diagnose_ex_tiling.py --ckpt /workspace/models/ex_detector_seed7/ex_detector_best.pth --stem IDRiD_59
"""
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--stem", required=True)
p.add_argument("--patch", type=int, default=512)
p.add_argument("--stride", type=int, default=384)
p.add_argument("--out", default=None)
p.add_argument("--drive_dir", default=None)
args = p.parse_args()

BASE   = Path("/workspace/data/idrid/A. Segmentation")
IMAGES = BASE / "1. Original Images" / "b. Testing Set"
GT_ROOT = BASE / "2. All Segmentation Groundtruths" / "b. Testing Set" / "3. Hard Exudates"
OUT = Path(args.out) if args.out else Path(f"/workspace/{args.stem}_tiling_diagnosis.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()
print(f"Loaded checkpoint: epoch={ckpt.get('epoch')} seed={ckpt.get('seed')}")


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


img_bgr = cv2.imread(str(IMAGES / f"{args.stem}.jpg"))
img_rgb = preprocess(img_bgr)
H, W = img_rgb.shape[:2]

gt_path = next((GT_ROOT / f"{args.stem}_EX{ext}" for ext in ('.tif', '.png', '.bmp')
                if (GT_ROOT / f"{args.stem}_EX{ext}").exists()), None)
gt = np.zeros((H, W), dtype=np.uint8)
if gt_path:
    gm = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gm is not None:
        gt = (gm > 0).astype(np.uint8) * 255

patch, stride = args.patch, args.stride
prob_map = np.zeros((H, W), dtype=np.float32)
count = np.zeros((H, W), dtype=np.float32)
ys = list(range(0, H - patch, stride)) + [H - patch]
xs = list(range(0, W - patch, stride)) + [W - patch]

# Also keep individual patch predictions so we can inspect boundary disagreement
patch_boxes = []
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
            patch_boxes.append((x0c, y0c, x0c+patch, y0c+patch))

prob_map = prob_map / np.maximum(count, 1)
pred_bin = (prob_map > 0.35).astype(np.uint8)

# Find the largest predicted blob (likely candidate for the suspicious one)
from skimage import measure as sk_measure
labeled = sk_measure.label(pred_bin)
regions = sk_measure.regionprops(labeled)
regions.sort(key=lambda r: -r.area)

fig, axes = plt.subplots(1, 4, figsize=(28, 7))

axes[0].imshow(img_rgb)
axes[0].set_title(f"{args.stem} — original")
axes[0].axis("off")

axes[1].imshow(prob_map, cmap="jet", vmin=0, vmax=1)
for (x0, y0, x1, y1) in patch_boxes:
    axes[1].add_patch(plt.Rectangle((x0, y0), patch, patch, fill=False,
                                     edgecolor="white", linewidth=0.5, alpha=0.4))
axes[1].set_title("Raw probability map + patch grid overlay\n(white lines = sliding-window boundaries)")
axes[1].axis("off")

axes[2].imshow(count, cmap="viridis")
axes[2].set_title("Overlap count map\n(how many patches covered each pixel)")
axes[2].axis("off")

overlay = img_rgb.copy()
gt_b = gt > 0
pred_b = pred_bin > 0
overlay[gt_b] = (0.4*overlay[gt_b] + 0.6*np.array([0,255,0])).astype(np.uint8)
overlay[pred_b & ~gt_b] = (0.4*overlay[pred_b & ~gt_b] + 0.6*np.array([255,0,255])).astype(np.uint8)
axes[3].imshow(overlay)
axes[3].set_title(f"GT (green) vs prediction FP (magenta)\n{len(regions)} predicted blobs, "
                   f"largest={regions[0].area if regions else 0}px")
axes[3].axis("off")

# Zoom in on the largest blob if it's suspiciously large/rectangular
if regions:
    r0, c0, r1, c1 = regions[0].bbox
    bw, bh = c1 - c0, r1 - r0
    extent = bw * bh
    fill_ratio = regions[0].area / max(extent, 1)
    print(f"Largest blob: bbox=({c0},{r0},{c1},{r1})  size={bw}x{bh}  "
          f"area={regions[0].area}  bbox_fill_ratio={fill_ratio:.3f}  "
          f"(near 1.0 = suspiciously rectangular/blocky)")

plt.tight_layout()
plt.savefig(str(OUT), dpi=100, bbox_inches="tight")
print(f"Saved -> {OUT}")

if args.drive_dir:
    import subprocess as _sp
    r = _sp.run(["rclone", "copy", str(OUT), args.drive_dir], capture_output=True, text=True)
    print("Drive backup:", "OK" if r.returncode == 0 else r.stderr.strip())
