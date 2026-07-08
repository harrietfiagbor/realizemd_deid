"""
viz_ma_learned.py

Visualizes the learned MA detector (run1, epoch 15) vs rule-based MA on IDRiD test images.

5 panels per image:
  GT | Rule-based baseline | Learned (epoch 15) | FP diff | Prob heatmap

Output: MA_learned_vs_rulebased_viz.png

Run:
    conda run -n py311 python viz_ma_learned.py
"""
import sys
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage import measure as sk_measure

BASE    = Path("/workspace/data/idrid/A. Segmentation")
IMAGES  = BASE / "1. Original Images" / "b. Testing Set"
GT_ROOT = BASE / "2. All Segmentation Groundtruths" / "b. Testing Set" / "1. Microaneurysms"
CKPT    = Path("/workspace/models/ma_detector/ma_detector_best.pth")
OUT     = Path("/workspace/MA_learned_vs_rulebased_viz.png")

sys.path.insert(0, "/workspace/realizemd_deid")
from pipeline.pathology import detect_optic_disc

STEMS     = ["IDRiD_55", "IDRiD_59", "IDRiD_64"]
PATCH     = 512
STRIDE    = 384
THRESHOLD = 0.35
S         = 900


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def detect_ma_baseline(img_rgb, od):
    green = img_rgb[:, :, 1]
    smoothed = cv2.GaussianBlur(green, (3, 3), 1.0)
    bg = cv2.medianBlur(green, 25)
    shade = np.clip(bg.astype(np.int16) - smoothed.astype(np.int16), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(shade, 12, 255, cv2.THRESH_BINARY)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(od))
    labeled = sk_measure.label(mask > 0)
    out = np.zeros_like(mask)
    for r in sk_measure.regionprops(labeled):
        if 20 <= r.area <= 1500:
            out[labeled == r.label] = 255
    return out


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
print(f"Model loaded: epoch={ckpt.get('epoch')}  recall={ckpt.get('recall',0):.4f}  "
      f"preservation={ckpt.get('preservation',0):.4f}")

n = len(STEMS)
fig, axes = plt.subplots(n, 5, figsize=(32, 6.5 * n))
fig.suptitle(
    "MA (Microaneurysms): Rule-based vs Learned (run1, epoch 15)\n"
    "GT  |  Rule-based baseline  |  Learned  |  FP comparison  |  Learned probability heatmap",
    fontsize=13, fontweight='bold', y=1.01)

for row, stem in enumerate(STEMS):
    print(f"  {stem}...")
    img_bgr = cv2.imread(str(IMAGES / f"{stem}.jpg"))
    img_rgb = preprocess(img_bgr)
    od = detect_optic_disc(img_rgb)

    gt_path = next((GT_ROOT / f"{stem}_MA{ext}"
                    for ext in ('.tif', '.png', '.bmp')
                    if (GT_ROOT / f"{stem}_MA{ext}").exists()), None)
    gt_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    if gt_path:
        gm = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gm is not None:
            gt_mask = (gm > 0).astype(np.uint8) * 255

    pred_base = detect_ma_baseline(img_rgb, od)
    prob = get_prob_map(model, img_rgb, device)
    pred_learned = ((prob > THRESHOLD) * 255).astype(np.uint8)

    gt_bin = (gt_mask > 0).astype(np.uint8)
    b_bin = (pred_base > 0).astype(np.uint8)
    l_bin = (pred_learned > 0).astype(np.uint8)

    rec_b, pre_b = blob_recall(gt_bin, b_bin)[0], pixel_precision(gt_bin, b_bin)
    rec_l, pre_l = blob_recall(gt_bin, l_bin)[0], pixel_precision(gt_bin, l_bin)

    img_s = resize_s(img_rgb)
    gt_s = resize_s(gt_mask)
    b_s = resize_s(pred_base)
    l_s = resize_s(pred_learned)
    gt_sb = gt_s > 0

    diff = img_s.copy()
    b_fp = (b_s > 0) & ~gt_sb
    l_fp = (l_s > 0) & ~gt_sb
    diff[gt_sb] = (0.4 * diff[gt_sb] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
    diff[b_fp] = (0.4 * diff[b_fp] + 0.6 * np.array([0, 230, 230])).astype(np.uint8)
    diff[l_fp] = (0.4 * diff[l_fp] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)

    heatmap = cv2.applyColorMap((resize_s((prob * 255).astype(np.uint8))), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    panels = [
        (overlay(img_s, gt_s, [0, 200, 0]),
         f"{stem}\nGT ({int(gt_bin.sum()/1000)}k px)"),
        (overlay(img_s, b_s, [0, 230, 230]),
         f"Rule-based\nrec={rec_b:.2f}  pre={pre_b:.2f}"),
        (overlay(img_s, l_s, [255, 0, 255]),
         f"Learned (epoch 15)\nrec={rec_l:.2f}  pre={pre_l:.2f}"),
        (diff,
         "FP comparison\ncyan=rule-based FP  magenta=learned FP  green=GT"),
        (heatmap,
         "Learned probability heatmap\n(red=high confidence)"),
    ]

    for col, (panel, title) in enumerate(panels):
        ax = axes[row, col] if n > 1 else axes[col]
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(title, fontsize=8, pad=4)

patches = [
    mpatches.Patch(color='green', label='GT lesion'),
    mpatches.Patch(color='cyan', label='Rule-based FP'),
    mpatches.Patch(color='magenta', label='Learned FP'),
]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(str(OUT), dpi=85, bbox_inches='tight')
print(f"Saved -> {OUT}")
plt.close()
