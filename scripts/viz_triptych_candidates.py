#!/usr/bin/env python3
"""
viz_triptych_candidates.py

Visual sanity check on Stage 2's top-ranked triptych candidates, before
handing any list to Harriet. High blob-count/area scores can mean genuine
severe pathology OR detector confusion on noise/artifacts/haze -- this
can't be told apart from the numbers alone, so render the actual overlays.

Usage (pod):
    python realizemd_deid/scripts/viz_triptych_candidates.py \
        --pool_dir /workspace/triptych/stage1_pool \
        --ranked_csv /workspace/triptych/triptych_candidates_stage2_ranked.csv \
        --ex_ckpt /workspace/models/ex_detector/ex_detector_best.pth \
        --he_ckpt /workspace/models/he_detector/he_detector_best.pth \
        --out_dir /workspace/triptych \
        --top_n 20
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

COLOR_EX = [30, 60, 220]
COLOR_HE = [220, 30, 150]
ROWS_PER_PLOT = 10


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def get_prob_map(model, img_rgb, device, patch, stride):
    H, W = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - patch, stride)) + [H - patch] if H > patch else [0]
    xs = list(range(0, W - patch, stride)) + [W - patch] if W > patch else [0]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0c, x0c = max(0, min(y0, max(H - patch, 0))), max(0, min(x0, max(W - patch, 0)))
                crop = img_rgb[y0c:y0c+patch, x0c:x0c+patch]
                if crop.shape[0] != patch or crop.shape[1] != patch:
                    crop = cv2.resize(crop, (patch, patch))
                t = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                prob = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                ph, pw = min(patch, H - y0c), min(patch, W - x0c)
                prob_map[y0c:y0c+ph, x0c:x0c+pw] += prob[:ph, :pw]
                count[y0c:y0c+ph, x0c:x0c+pw] += 1
    return prob_map / np.maximum(count, 1)


def overlay(img_rgb, mask, color, alpha=0.45):
    out = img_rgb.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def resize_s(img, s=500):
    h, w = img.shape[:2]
    scale = s / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ex_ckpt = torch.load(args.ex_ckpt, map_location="cpu", weights_only=False)
    ex_model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1).to(device)
    ex_model.load_state_dict(ex_ckpt["state_dict"])
    ex_model.eval()

    he_ckpt = torch.load(args.he_ckpt, map_location="cpu", weights_only=False)
    he_model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1).to(device)
    he_model.load_state_dict(he_ckpt["state_dict"])
    he_model.eval()

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.ranked_csv) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            rows.append(parts)
    rows = rows[args.offset: args.offset + args.top_n]

    n_parts = (len(rows) + ROWS_PER_PLOT - 1) // ROWS_PER_PLOT
    for part in range(n_parts):
        chunk = rows[part * ROWS_PER_PLOT: (part + 1) * ROWS_PER_PLOT]
        n = len(chunk)
        fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
        fig.suptitle(f"Triptych Candidates -- Visual Sanity Check -- part {part + 1}/{n_parts}\n"
                     "Original | EX overlay (blue) | HE overlay (magenta)",
                     fontsize=12, fontweight="bold", y=1.01)

        for row, r in enumerate(chunk):
            stem, ex_n, ex_area, he_n, he_area, score = r
            img_bgr = cv2.imread(str(pool_dir / f"{stem}.jpeg"))
            img_rgb = preprocess(img_bgr)

            ex_prob = get_prob_map(ex_model, img_rgb, device, 512, 384)
            ex_bin = ((ex_prob > 0.35).astype(np.uint8)) * 255
            he_prob = get_prob_map(he_model, img_rgb, device, 768, 640)
            he_bin = ((he_prob > 0.35).astype(np.uint8)) * 255

            img_s, ex_s, he_s = resize_s(img_rgb), resize_s(ex_bin), resize_s(he_bin)

            panels = [
                (img_s, f"{stem}\nscore={float(score):.1f}"),
                (overlay(img_s, ex_s, COLOR_EX), f"EX: {ex_n} blobs, {ex_area}px"),
                (overlay(img_s, he_s, COLOR_HE), f"HE: {he_n} blobs, {he_area}px"),
            ]
            for col, (panel, title) in enumerate(panels):
                ax = axes[row, col] if n > 1 else axes[col]
                ax.imshow(panel)
                ax.axis("off")
                ax.set_title(title, fontsize=9)

        plt.tight_layout()
        out_png = out_dir / f"triptych_candidates_viz_offset{args.offset}_part{part + 1}.png"
        plt.savefig(str(out_png), dpi=80, bbox_inches="tight")
        print(f"Saved -> {out_png}")
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool_dir", required=True)
    p.add_argument("--ranked_csv", required=True)
    p.add_argument("--ex_ckpt", required=True)
    p.add_argument("--he_ckpt", required=True)
    p.add_argument("--out_dir", default="triptych")
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--offset", type=int, default=0,
                    help="Skip the first N ranked candidates (e.g. 20 to review ranks 21-40)")
    main(p.parse_args())
