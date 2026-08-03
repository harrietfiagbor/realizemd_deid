#!/usr/bin/env python3
"""
eval_od_learned_resolution.py

Resolution-robustness check for the learned OD detector, matching the
methodology already used for the brightness-heuristic version
(eval_od_resolution.py) and for EX/HE: resize the INPUT image first
(reproducing what the production pipeline actually does), then detect.

Unlike the heuristic script, this only tests [512, 256], not
[fullres, 512, 256]. Reason: the model always resizes its input to
img_size (512) internally before its forward pass, so "fullres" and
"512" are mathematically identical for this architecture -- both reduce
to a single resize-to-512 operation. The only real question is whether
pre-resizing to 256 (the project's now-locked production resolution)
before the model's own upsample-back-to-512 step loses enough detail to
hurt accuracy compared to a native 512 pipeline.

For each IDRiD-test image, at each of [512, 256]:
  - resize the ORIGINAL image to that size first
  - CLAHE (matches training preprocessing exactly, no FOV masking --
    the model was trained without it, so this evaluates the model as
    actually trained, not a hypothetical integrated-pipeline variant)
  - resize to img_size (512) for the model's forward pass (no-op when
    target is already 512)
  - resize GT to target size via maxpool (matches eval_ex_256px.py /
    eval_od_resolution.py convention)
  - compute Dice/IoU

Usage (pod):
    python realizemd_deid/scripts/eval_od_learned_resolution.py \
        --idrid_dir "/workspace/data/idrid/A. Segmentation" \
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
import torch.nn.functional as F


def dice_coef(pred_bin, gt_bin):
    inter = int((pred_bin & gt_bin).sum())
    denom = int(pred_bin.sum()) + int(gt_bin.sum())
    return 1.0 if denom == 0 else (2.0 * inter) / denom


def iou_score(pred_bin, gt_bin):
    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    return 1.0 if union == 0 else inter / union


def clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def resize_gt_maxpool(gt_bin, target_hw):
    x = torch.from_numpy(gt_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.adaptive_max_pool2d(x, target_hw)
    return (out.squeeze().numpy() > 0.5).astype(np.uint8)


def overlay(img_rgb, mask, color, alpha=0.4):
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

    idrid_dir = Path(args.idrid_dir)
    images_dir = idrid_dir / "1. Original Images" / "b. Testing Set"
    gt_dir = idrid_dir / "2. All Segmentation Groundtruths" / "b. Testing Set" / "5. Optic Disc"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    RESOLUTIONS = [512, 256]
    VIZ_STEMS = ["IDRiD_55", "IDRiD_60", "IDRiD_65"]
    results = {r: {"dice": [], "iou": []} for r in RESOLUTIONS}
    viz_data = {s: {} for s in VIZ_STEMS}

    log_path = out_dir / "eval_od_learned_resolution_results.txt"
    log = open(log_path, "w", buffering=1)

    def p(msg=""):
        log.write(msg + "\n")
        print(msg)

    p("OD learned-detector resolution robustness -- resizing INPUT before detection")
    p("(fullres omitted: identical to 512 for this whole-image-resize architecture)")
    p("")

    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
        gt_path = next((gt_dir / f"{stem}_OD{ext}" for ext in (".tif", ".png", ".bmp")
                        if (gt_dir / f"{stem}_OD{ext}").exists()), None)
        if gt_path is None:
            continue
        img_bgr = cv2.imread(str(img_path))
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_raw is None or gt_raw.max() == 0:
            continue
        img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_bin_full = (gt_raw > 0).astype(np.uint8)

        p(stem)
        for res in RESOLUTIONS:
            img_res = cv2.resize(img_rgb_full, (res, res))
            img_clahe = clahe_rgb(img_res)
            img_model_in = cv2.resize(img_clahe, (img_size, img_size)) if res != img_size else img_clahe

            with torch.no_grad():
                t = torch.from_numpy(img_model_in.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                prob = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
            pred_bin_model = (prob > 0.5).astype(np.uint8)
            pred_bin = cv2.resize(pred_bin_model, (res, res), interpolation=cv2.INTER_NEAREST) if res != img_size else pred_bin_model

            gt_bin = resize_gt_maxpool(gt_bin_full, (res, res))

            d = dice_coef(pred_bin, gt_bin)
            i = iou_score(pred_bin, gt_bin)
            results[res]["dice"].append(d)
            results[res]["iou"].append(i)
            p(f"  {res:4d}px  dice={d:.4f}  iou={i:.4f}")

            if stem in VIZ_STEMS:
                viz_data[stem][res] = {"img": img_res, "gt": gt_bin, "pred": pred_bin, "dice": d, "iou": i}

    p(f"\n{'='*50}")
    p("SUMMARY")
    p(f"{'='*50}")
    for res in RESOLUTIONS:
        r = results[res]
        p(f"{res:4d}px  mean_dice={np.mean(r['dice']):.4f}  mean_iou={np.mean(r['iou']):.4f}  n={len(r['dice'])}")
    p("\nDone.")
    log.close()

    # Visualization
    n_stems, n_res = len(VIZ_STEMS), len(RESOLUTIONS)
    fig, axes = plt.subplots(n_stems, n_res, figsize=(6 * n_res, 6 * n_stems))
    fig.suptitle("Learned OD Detector: Resolution Robustness\n"
                 "GT (yellow) vs predicted mask (cyan) -- input resized BEFORE detection",
                 fontsize=12, fontweight="bold", y=1.02)
    for row, stem in enumerate(VIZ_STEMS):
        for col, res in enumerate(RESOLUTIONS):
            d = viz_data[stem][res]
            panel = overlay(d["img"], d["gt"], [255, 255, 0], alpha=0.3)
            panel = overlay(panel, d["pred"], [0, 220, 220], alpha=0.4)
            ax = axes[row, col] if n_stems > 1 else axes[col]
            ax.imshow(panel)
            ax.axis("off")
            ax.set_title(f"{stem} @ {res}px\ndice={d['dice']:.3f}  iou={d['iou']:.3f}", fontsize=9)
    plt.tight_layout()
    viz_out = out_dir / "OD_learned_resolution_viz.png"
    plt.savefig(str(viz_out), dpi=100, bbox_inches="tight")
    print(f"Saved -> {viz_out}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--idrid_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", default="od_eval")
    main(p.parse_args())
