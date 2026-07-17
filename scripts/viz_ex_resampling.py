"""
viz_ex_resampling.py

Visualizes learned EX masks at full-res vs 256px vs 512px (maxpool).
Confirms whether EX blobs survive downsampling the way learned MA does,
unlike the rule-based EX detector's sub-pixel collapse at 256px.

6 panels per image:
  Full-res GT | Full-res prediction | 256px GT | 256px prediction | 512px GT | 512px prediction

Output: <ckpt-parent-dirname>_resampling_viz.png (or --out to override)

Run (on pod, GPU):
    python viz_ex_resampling.py --ckpt /workspace/models/ex_detector_seed7/ex_detector_best.pth
"""
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage import measure as sk_measure

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="/workspace/models/ex_detector/ex_detector_best.pth")
p.add_argument("--out", default=None)
p.add_argument("--drive_dir", default=None)
args = p.parse_args()

BASE    = Path("/workspace/data/idrid/A. Segmentation")
IMAGES  = BASE / "1. Original Images" / "b. Testing Set"
GT_ROOT = BASE / "2. All Segmentation Groundtruths" / "b. Testing Set" / "3. Hard Exudates"
CKPT    = Path(args.ckpt)
OUT     = Path(args.out) if args.out else Path(f"/workspace/{CKPT.parent.name}_resampling_viz.png")

STEMS     = ["IDRiD_55", "IDRiD_59", "IDRiD_64"]
PATCH     = 512
STRIDE    = 384
THRESHOLD = 0.35
S         = 700


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def get_prob_map(model, img_rgb, device):
    H, W = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - PATCH, STRIDE)) + [H - PATCH]
    xs = list(range(0, W - PATCH, STRIDE)) + [W - PATCH]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0 = max(0, min(y0, H - PATCH))
                x0 = max(0, min(x0, W - PATCH))
                crop = img_rgb[y0:y0+PATCH, x0:x0+PATCH]
                t = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                p = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0:y0+PATCH, x0:x0+PATCH] += p
                count[y0:y0+PATCH, x0:x0+PATCH] += 1
    return prob_map / np.maximum(count, 1)


def resample_maxpool(mask_np, out_hw):
    x = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.adaptive_max_pool2d(x, out_hw)
    return (out.squeeze().numpy() > 0.5).astype(np.uint8)


def overlay(img_rgb, mask, color, alpha=0.6):
    out = img_rgb.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def resize_s(img, s=S):
    h, w = img.shape[:2]
    scale = s / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def pixel_precision(gt_bin, pred_bin):
    pp = int((pred_bin > 0).sum())
    if pp == 0:
        return 1.0
    return int(((gt_bin > 0) & (pred_bin > 0)).sum()) / pp


def blob_recall(gt_bin, pred_bin, iou_thresh=0.3):
    gt_lab = sk_measure.label(gt_bin)
    pr_lab = sk_measure.label(pred_bin)
    pr_areas = {r.label: r.area for r in sk_measure.regionprops(pr_lab)}
    hits = n_gt = 0
    for gr in sk_measure.regionprops(gt_lab):
        n_gt += 1
        r0, c0, r1, c1 = gr.bbox
        gb = (gt_lab[r0:r1, c0:c1] == gr.label).astype(np.uint8)
        pc = pr_lab[r0:r1, c0:c1]
        for pl in np.unique(pc[gb > 0]):
            if pl == 0:
                continue
            inter = int((gb & (pc == pl)).sum())
            union = gr.area + pr_areas.get(pl, 0) - inter
            if inter / max(union, 1) >= iou_thresh:
                hits += 1
                break
    return hits / max(n_gt, 1), hits, n_gt


device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()
print(f"Model loaded (epoch={ckpt.get('epoch')}, recall={ckpt.get('recall',0):.4f})")

n = len(STEMS)
fig, axes = plt.subplots(n, 6, figsize=(36, 6 * n))
fig.suptitle(
    "EX (Hard Exudates) — Full res vs 256px vs 512px (maxpool), learned detector\n"
    "Full-res GT | Full-res pred | 256px GT | 256px pred | 512px GT | 512px pred",
    fontsize=12, fontweight='bold', y=1.01)

for row, stem in enumerate(STEMS):
    print(f"  {stem}...")
    img_bgr = cv2.imread(str(IMAGES / f"{stem}.jpg"))
    img_rgb = preprocess(img_bgr)

    gt_path = next((GT_ROOT / f"{stem}_EX{ext}"
                    for ext in ('.tif', '.png', '.bmp')
                    if (GT_ROOT / f"{stem}_EX{ext}").exists()), None)
    gt_full = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    if gt_path:
        gm = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gm is not None:
            gt_full = (gm > 0).astype(np.uint8) * 255

    prob = get_prob_map(model, img_rgb, device)
    pred_full = ((prob > THRESHOLD) * 255).astype(np.uint8)

    gt_256 = resample_maxpool((gt_full > 0).astype(np.uint8), (256, 256))
    pred_256 = resample_maxpool((pred_full > 0).astype(np.uint8), (256, 256))
    gt_512 = resample_maxpool((gt_full > 0).astype(np.uint8), (512, 512))
    pred_512 = resample_maxpool((pred_full > 0).astype(np.uint8), (512, 512))

    rec_fr, _, n_fr = blob_recall((gt_full > 0).astype(np.uint8), (pred_full > 0).astype(np.uint8))
    pre_fr = pixel_precision((gt_full > 0).astype(np.uint8), (pred_full > 0).astype(np.uint8))
    rec_256, _, n_256 = blob_recall(gt_256, pred_256)
    pre_256 = pixel_precision(gt_256, pred_256)
    rec_512, _, n_512 = blob_recall(gt_512, pred_512)
    pre_512 = pixel_precision(gt_512, pred_512)

    img_s = resize_s(img_rgb)
    gt_s = resize_s(gt_full)
    pred_s = resize_s(pred_full)

    img_256_disp = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    gt_256_disp = cv2.resize(gt_256 * 255, (256, 256), interpolation=cv2.INTER_NEAREST)
    pred_256_disp = cv2.resize(pred_256 * 255, (256, 256), interpolation=cv2.INTER_NEAREST)

    img_512_disp = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    gt_512_disp = cv2.resize(gt_512 * 255, (512, 512), interpolation=cv2.INTER_NEAREST)
    pred_512_disp = cv2.resize(pred_512 * 255, (512, 512), interpolation=cv2.INTER_NEAREST)

    panels = [
        (overlay(img_s, gt_s, [255, 255, 0]),
         f"{stem} — Full-res GT\n({n_fr} blobs)"),
        (overlay(img_s, pred_s, [255, 0, 255]),
         f"Full-res pred\nrec={rec_fr:.2f}  pre={pre_fr:.2f}"),
        (overlay(img_256_disp, gt_256_disp, [255, 255, 0]),
         f"256px GT (maxpool)\n({n_256} blobs)"),
        (overlay(img_256_disp, pred_256_disp, [255, 0, 255]),
         f"256px pred\nrec={rec_256:.2f}  pre={pre_256:.2f}"),
        (overlay(img_512_disp, gt_512_disp, [255, 255, 0]),
         f"512px GT (maxpool)\n({n_512} blobs)"),
        (overlay(img_512_disp, pred_512_disp, [255, 0, 255]),
         f"512px pred\nrec={rec_512:.2f}  pre={pre_512:.2f}"),
    ]

    for col, (panel, title) in enumerate(panels):
        ax = axes[row, col] if n > 1 else axes[col]
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(title, fontsize=8, pad=4)

patches = [
    mpatches.Patch(color='yellow', label='GT lesion'),
    mpatches.Patch(color='magenta', label='Prediction'),
]
fig.legend(handles=patches, loc='lower center', ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(str(OUT), dpi=85, bbox_inches='tight')
print(f"Saved -> {OUT}")
plt.close()

if args.drive_dir:
    import subprocess as _sp
    r = _sp.run(["rclone", "copy", str(OUT), args.drive_dir], capture_output=True, text=True)
    print("Drive backup:", "OK" if r.returncode == 0 else r.stderr.strip())
