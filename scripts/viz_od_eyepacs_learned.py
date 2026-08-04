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
from skimage import measure as sk_measure

RESOLUTIONS = [512, 256]
ROWS_PER_PLOT = 10


def blob_metrics(pred_bin):
    """Objective, auditable signals that don't require ground truth:
    blob count, largest-blob coverage/centroid/border-distance. A disc is
    never at the frame edge, so border-distance is a real tell for the
    edge/glare-crescent wrong-blob failure mode specifically."""
    H, W = pred_bin.shape
    labeled = sk_measure.label(pred_bin)
    regions = sk_measure.regionprops(labeled)
    n_blobs = len(regions)
    if n_blobs == 0:
        return n_blobs, 0.0, None, None, None
    largest = max(regions, key=lambda r: r.area)
    cy, cx = largest.centroid
    cy_norm, cx_norm = cy / H, cx / W
    r0, c0, r1, c1 = largest.bbox
    border_dist_px = min(r0, c0, H - r1, W - c1)
    border_dist_frac = border_dist_px / min(H, W)
    return n_blobs, 100.0 * largest.area / (H * W), (cy_norm, cx_norm), border_dist_frac, int(largest.area)


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
    log_lines = ["filename,dr_grade,verdict_512,verdict_256,notes"]
    data_lines = ["filename,dr_grade,resolution,coverage_pct,n_blobs,"
                  "largest_blob_area_px,centroid_y_norm,centroid_x_norm,border_dist_frac"]

    n_parts = (len(stems) + ROWS_PER_PLOT - 1) // ROWS_PER_PLOT
    for part in range(n_parts):
        chunk = stems[part * ROWS_PER_PLOT: (part + 1) * ROWS_PER_PLOT]
        n = len(chunk)
        fig, axes = plt.subplots(n, n_cols, figsize=(6 * n_cols, 5 * n))
        fig.suptitle(f"Learned OD Detector on EyePACS (same 30 images as the heuristic check, "
                     f"no ground truth -- visual review only) -- part {part + 1}/{n_parts}\n"
                     "Original | Overlay @512px | Overlay @256px  --  cyan = predicted disc region",
                     fontsize=12, fontweight="bold", y=1.01)

        for row, stem in enumerate(chunk):
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

                n_blobs, coverage, centroid, border_dist, largest_area = blob_metrics(pred_bin > 0)
                coverages.append(coverage)
                panels.append(overlay(img_res, pred_bin, [0, 220, 220]))

                cy_s = f"{centroid[0]:.3f}" if centroid else ""
                cx_s = f"{centroid[1]:.3f}" if centroid else ""
                bd_s = f"{border_dist:.3f}" if border_dist is not None else ""
                data_lines.append(f"{stem},{grade},{res},{coverage:.3f},{n_blobs},"
                                   f"{largest_area or 0},{cy_s},{cx_s},{bd_s}")

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
        out_png = out_dir / f"OD_learned_eyepacs_viz_part{part + 1}.png"
        plt.savefig(str(out_png), dpi=80, bbox_inches="tight")
        print(f"Saved -> {out_png}")
        plt.close()

    out_log = out_dir / "OD_learned_eyepacs_viz_log.txt"
    with open(out_log, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Saved -> {out_log} (fill in verdict_512/verdict_256: correct / oversized / wrong-blob / other)")

    out_data = out_dir / "OD_learned_eyepacs_viz_data.csv"
    with open(out_data, "w") as f:
        f.write("\n".join(data_lines) + "\n")
    print(f"Saved -> {out_data} (objective per-image metrics: coverage, blob count, "
          f"centroid position, border distance -- backs the visual verdicts with numbers)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", default="od_eval")
    main(p.parse_args())
