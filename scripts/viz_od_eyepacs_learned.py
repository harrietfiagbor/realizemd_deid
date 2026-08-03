#!/usr/bin/env python3
"""
viz_od_eyepacs_learned.py

Re-runs the EyePACS qualitative check (originally viz_od_eyepacs.py,
which found the brightness heuristic wrong-blob rate of ~33%) against the
learned OD detector instead, on the EXACT SAME 30 stratified images
(OD_eyepacs_viz_log.txt) for a direct before/after comparison -- not a
fresh/different sample.

No ground truth exists for EyePACS OD location, so this produces a
review gallery for manual inspection, not an automated metric -- same
as the original.

Usage (pod):
    python realizemd_deid/scripts/viz_od_eyepacs_learned.py \
        --sample_dir /workspace/od_detector/data/eyepacs_od_sample \
        --checkpoint /workspace/od_detector/models/od_run1_b0/od_detector_best.pth \
        --out_dir /workspace/od_detector/eval
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

RESOLUTIONS = [512, 256]


def clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def overlay(img_rgb, mask, color, alpha=0.45):
    out = img_rgb.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name=ckpt["encoder"], in_channels=3, classes=1).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt["img_size"]
    print(f"Checkpoint: epoch={ckpt['epoch']} encoder={ckpt['encoder']} img_size={img_size}")

    sample_dir = Path(args.sample_dir)
    log_path = sample_dir / "OD_eyepacs_viz_log.txt"
    stems, grades = [], {}
    with open(log_path) as f:
        next(f)
        for line in f:
            stem, grade = line.strip().split(",")[:2]
            stems.append(stem)
            grades[stem] = grade

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 1 + len(RESOLUTIONS)
    n = len(stems)
    fig, axes = plt.subplots(n, n_cols, figsize=(6 * n_cols, 5 * n))
    fig.suptitle("Learned OD Detector on EyePACS (same 30 images as the heuristic check, "
                 "no ground truth -- visual review only)\n"
                 "Original | Overlay @512px | Overlay @256px  --  cyan = predicted disc region",
                 fontsize=12, fontweight="bold", y=1.005)

    log_lines = ["filename,dr_grade,verdict_512,verdict_256,notes"]

    for row, stem in enumerate(stems):
        grade = grades[stem]
        img_path = sample_dir / f"{stem}.jpeg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: could not read {img_path}")
            continue
        img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        panels, coverages = [], []
        for res in RESOLUTIONS:
            img_res = cv2.resize(img_rgb_full, (res, res))
            img_clahe = clahe_rgb(img_res)
            img_model_in = cv2.resize(img_clahe, (img_size, img_size)) if res != img_size else img_clahe

            with torch.no_grad():
                t = torch.from_numpy(img_model_in.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                prob = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
            pred_bin_model = (prob > 0.5).astype(np.uint8)
            pred_bin = cv2.resize(pred_bin_model, (res, res), interpolation=cv2.INTER_NEAREST) if res != img_size else pred_bin_model

            coverage = 100.0 * pred_bin.sum() / (res * res)
            coverages.append(coverage)
            panels.append(overlay(img_res, pred_bin, [0, 220, 220]))

        orig_disp = cv2.resize(img_rgb_full, (400, 400))
        row_panels = [(orig_disp, f"{stem}\n(grade {grade})")] + [
            (panels[i], f"@{RESOLUTIONS[i]}px  coverage={coverages[i]:.1f}%")
            for i in range(len(RESOLUTIONS))
        ]
        for col, (panel, title) in enumerate(row_panels):
            ax = axes[row, col] if n > 1 else axes[col]
            ax.imshow(panel)
            ax.axis("off")
            ax.set_title(title, fontsize=9)

        log_lines.append(f"{stem},{grade},,,")

    plt.tight_layout()
    out_png = out_dir / "OD_learned_eyepacs_viz.png"
    plt.savefig(str(out_png), dpi=70, bbox_inches="tight")
    print(f"Saved -> {out_png}")
    plt.close()

    out_log = out_dir / "OD_learned_eyepacs_viz_log.txt"
    with open(out_log, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Saved -> {out_log} (fill in verdict_512/verdict_256: correct / oversized / wrong-blob / other)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", default="od_eval")
    main(p.parse_args())
