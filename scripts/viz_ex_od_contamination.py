#!/usr/bin/env python3
"""
viz_ex_od_contamination.py

Visualizes EX disc-contamination before/after learned-OD-mask exclusion, on
the worst contaminated images found by check_od_contamination_learned.py
(the >500px-on-disc cases), so the numeric result is visually verifiable
rather than trusted on the summary print alone.

Panels per image: original | GT EX | raw EX prediction (disc outlined) |
EX prediction after learned-OD exclusion.

Usage (pod):
    python realizemd_deid/scripts/viz_ex_od_contamination.py \
        --od_ckpt /workspace/od_detector/models/od_run1_b0/od_detector_best.pth \
        --idrid_dir "/workspace/data/idrid/A. Segmentation" \
        --out_dir /workspace/od_detector/eval \
        --case seed7:/workspace/models/ex_detector_seed7/ex_detector_best.pth:IDRiD_59 \
        --case seed7:/workspace/models/ex_detector_seed7/ex_detector_best.pth:IDRiD_80 \
        --case seed123:/workspace/models/ex_detector_seed123/ex_detector_best.pth:IDRiD_64
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

COLOR_GT = [0, 180, 0]
COLOR_PRED = [30, 60, 220]
COLOR_DISC_OUTLINE = [255, 255, 0]


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def get_ex_prob_map(model, img_rgb, device, patch=512, stride=384):
    H, W = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - patch, stride)) + [H - patch]
    xs = list(range(0, W - patch, stride)) + [W - patch]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0c, x0c = max(0, min(y0, H - patch)), max(0, min(x0, W - patch))
                crop = img_rgb[y0c:y0c+patch, x0c:x0c+patch]
                t = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                prob = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0c:y0c+patch, x0c:x0c+patch] += prob
                count[y0c:y0c+patch, x0c:x0c+patch] += 1
    return prob_map / np.maximum(count, 1)


def get_od_mask(model, img_rgb, device, img_size):
    H, W = img_rgb.shape[:2]
    img_r = cv2.resize(img_rgb, (img_size, img_size))
    with torch.no_grad():
        t = torch.from_numpy(img_r.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        prob = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
    pred_small = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_NEAREST)


def overlay(img_rgb, mask, color, alpha=0.5):
    out = img_rgb.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def resize_s(img, s=700):
    h, w = img.shape[:2]
    scale = s / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    od_ckpt = torch.load(args.od_ckpt, map_location="cpu", weights_only=False)
    od_model = smp.Unet(encoder_name=od_ckpt["encoder"], in_channels=3, classes=1).to(device)
    od_model.load_state_dict(od_ckpt["state_dict"])
    od_model.eval()
    od_img_size = od_ckpt["img_size"]

    idrid_dir = Path(args.idrid_dir)
    images_dir = idrid_dir / "1. Original Images" / "b. Testing Set"
    gt_dir = idrid_dir / "2. All Segmentation Groundtruths" / "b. Testing Set" / "3. Hard Exudates"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ex_models = {}

    cases = []
    for c in args.case:
        label, ckpt_path, stem = c.split(":", 2)
        cases.append((label, ckpt_path, stem))

    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(24, 6 * n))
    fig.suptitle("EX Disc-Contamination: Before vs After Learned-OD Exclusion\n"
                 "Original | GT (green) | Raw EX pred (blue) + disc outline (yellow) | Excluded EX pred (blue)",
                 fontsize=12, fontweight="bold", y=1.01)

    for row, (label, ex_ckpt_path, stem) in enumerate(cases):
        if ex_ckpt_path not in ex_models:
            ckpt = torch.load(ex_ckpt_path, map_location="cpu", weights_only=False)
            m = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1).to(device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            ex_models[ex_ckpt_path] = m
        ex_model = ex_models[ex_ckpt_path]

        img_bgr = cv2.imread(str(images_dir / f"{stem}.jpg"))
        img_rgb = preprocess(img_bgr)
        gt_path = next(gt_dir.glob(f"{stem}_EX.*"), None)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path else np.zeros(img_rgb.shape[:2], np.uint8)
        gt_bin = (gt > 0).astype(np.uint8) * 255

        prob = get_ex_prob_map(ex_model, img_rgb, device)
        pred_raw = (prob > 0.35).astype(np.uint8) * 255
        od_mask = get_od_mask(od_model, img_rgb, device, od_img_size)
        pred_excluded = cv2.bitwise_and(pred_raw, cv2.bitwise_not(od_mask))

        fp_on_od = int(((pred_raw > 0) & (gt_bin == 0) & (od_mask > 0)).sum())
        fp_total = int(((pred_raw > 0) & (gt_bin == 0)).sum())

        img_s = resize_s(img_rgb)
        gt_s = resize_s(gt_bin)
        pred_raw_s = resize_s(pred_raw)
        pred_excl_s = resize_s(pred_excluded)
        od_s = resize_s(od_mask)

        od_contours, _ = cv2.findContours(od_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_panel = overlay(img_s, pred_raw_s, COLOR_PRED)
        cv2.drawContours(raw_panel, od_contours, -1, COLOR_DISC_OUTLINE, 2)

        panels = [
            (img_s, f"{label} / {stem}\nreal fundus photograph"),
            (overlay(img_s, gt_s, COLOR_GT), "GT hard exudates"),
            (raw_panel, f"Raw EX prediction + disc outline\n{fp_on_od}px FP on disc / {fp_total}px total FP"),
            (overlay(img_s, pred_excl_s, COLOR_PRED), "EX prediction after learned-OD exclusion"),
        ]
        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col] if n > 1 else axes[col]
            ax.imshow(panel)
            ax.axis("off")
            ax.set_title(title, fontsize=9, pad=4)

    plt.tight_layout()
    out_png = out_dir / "EX_OD_contamination_viz.png"
    plt.savefig(str(out_png), dpi=90, bbox_inches="tight")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--od_ckpt", required=True)
    p.add_argument("--idrid_dir", required=True)
    p.add_argument("--out_dir", default="od_eval")
    p.add_argument("--case", action="append", required=True,
                    help="label:ex_checkpoint_path:idrid_stem, repeatable")
    main(p.parse_args())
