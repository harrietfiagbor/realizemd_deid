"""
eval_ex_learned_256px.py

Tests learned EX (Hard Exudate) mask quality after downsampling to 256x256.
Uses the learned EX detector (run1, epoch 5, efficientnet-b2, threshold=0.35).
Answers Adam's question: does a learned EX detector survive 256px the way
learned MA did, potentially removing the argument for 512px as target resolution.

Checkpoint: models/ex_detector/ex_detector_best.pth (run1, epoch 5)

Output: eval_ex_learned_256px_results.txt

Run (on pod, GPU):
    python eval_ex_learned_256px.py
"""
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from pathlib import Path
from skimage import measure as sk_measure

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="/workspace/models/ex_detector/ex_detector_best.pth")
p.add_argument("--out", default=None)
p.add_argument("--drive_dir", default=None)
args = p.parse_args()

BASE    = Path("/workspace/data/idrid/A. Segmentation")
IMAGES  = BASE / "1. Original Images"
GT_ROOT = BASE / "2. All Segmentation Groundtruths"
CKPT    = Path(args.ckpt)
LOG     = Path(args.out) if args.out else Path(f"/workspace/eval_ex_learned_256px_{CKPT.parent.name}_results.txt")

TARGET_HW = (256, 256)
MODES     = ["maxpool", "maxpool+ero1", "maxpool+ero2", "maxpool+ero3", "area", "nearest"]
PATCH     = 512
STRIDE    = 384
THRESHOLD = 0.35

_log = open(str(LOG), "w", buffering=1)
def log(msg=""): _log.write(msg + "\n"); _log.flush(); print(msg)


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def get_prob_map(model, img_rgb, device):
    model.eval()
    H, W     = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, H - PATCH, STRIDE)) + [H - PATCH]
    xs = list(range(0, W - PATCH, STRIDE)) + [W - PATCH]
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0  = max(0, min(y0, H - PATCH))
                x0  = max(0, min(x0, W - PATCH))
                crop = img_rgb[y0:y0+PATCH, x0:x0+PATCH]
                t    = torch.from_numpy(crop.transpose(2, 0, 1)).float() / 255.0
                p    = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0:y0+PATCH, x0:x0+PATCH] += p
                count[y0:y0+PATCH, x0:x0+PATCH]    += 1
    return prob_map / np.maximum(count, 1)


def resample(mask_np: np.ndarray, out_hw: tuple, mode: str) -> np.ndarray:
    x = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if mode.startswith("maxpool"):
        out = F.adaptive_max_pool2d(x, out_hw)
    elif mode == "area":
        out = F.interpolate(x, size=out_hw, mode="area")
    elif mode == "nearest":
        out = F.interpolate(x, size=out_hw, mode="nearest")
    result = (out.squeeze().numpy() > 0.5).astype(np.uint8)
    if "+ero" in mode:
        ero_px = int(mode.split("ero")[1])
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_px * 2 + 1, ero_px * 2 + 1))
        result = cv2.erode(result, k)
    return result


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
            if pl == 0: continue
            inter = int((gb & (pc == pl)).sum())
            union = gr.area + pr_areas.get(pl, 0) - inter
            if inter / max(union, 1) >= iou_thresh:
                hits += 1; break
    return hits, n_gt


def pixel_precision(gt_bin, pred_bin):
    pp = int((pred_bin > 0).sum())
    if pp == 0: return 1.0
    return int(((gt_bin > 0) & (pred_bin > 0)).sum()) / pp


def eval_split(split_name, img_folder, gt_folder, model, device):
    log(f"\n{'='*70}")
    log(f"  {split_name}")
    log(f"{'='*70}")

    img_dir = IMAGES / img_folder
    gt_dir  = GT_ROOT / gt_folder / "3. Hard Exudates"
    img_paths = sorted(img_dir.glob("*.jpg"))

    keys   = ["fullres"] + MODES
    totals = {k: {"hits": 0, "gt": 0, "pp": []} for k in keys}

    log(f"\n{'image':16s} {'mode':12s} {'recall':>8s} {'precision':>10s} {'hits/gt':>10s}")
    log("-" * 60)

    for img_path in img_paths:
        stem = img_path.stem
        gt_path = next((gt_dir / f"{stem}_EX{ext}"
                        for ext in ('.tif', '.png', '.bmp')
                        if (gt_dir / f"{stem}_EX{ext}").exists()), None)
        if gt_path is None:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = preprocess(img_bgr)

        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None or gt_raw.max() == 0:
            continue

        prob = get_prob_map(model, img_rgb, device)
        gt_bin    = (gt_raw > 0).astype(np.uint8)
        pred_full = (prob > THRESHOLD).astype(np.uint8)

        h, n = blob_recall(gt_bin, pred_full)
        pp   = pixel_precision(gt_bin, pred_full)
        totals["fullres"]["hits"] += h
        totals["fullres"]["gt"]   += n
        totals["fullres"]["pp"].append(pp)
        log(f"{stem:16s} {'fullres':12s} {h/max(n,1):>8.3f} {pp:>10.3f} {h:>5}/{n}")

        gt_256 = resample(gt_bin, TARGET_HW, "maxpool")
        for mode in MODES:
            pred_256 = resample(pred_full, TARGET_HW, mode)
            h256, n256 = blob_recall(gt_256, pred_256)
            pp256      = pixel_precision(gt_256, pred_256)
            totals[mode]["hits"] += h256
            totals[mode]["gt"]   += n256
            totals[mode]["pp"].append(pp256)
            log(f"{'':16s} {mode:12s} {h256/max(n256,1):>8.3f} {pp256:>10.3f} {h256:>5}/{n256}")

    log(f"\n--- {split_name} Summary ---")
    log(f"{'mode':14s} {'recall':>10s} {'precision':>12s}  note")
    log("-" * 52)
    for k in keys:
        t = totals[k]
        rec = t["hits"] / max(t["gt"], 1)
        pre = float(np.mean(t["pp"])) if t["pp"] else 0.0
        note = "<-- full res reference" if k == "fullres" else ""
        log(f"{k:14s} {rec:>10.4f} {pre:>12.4f}  {note}")


if __name__ == "__main__":
    import time
    log("Learned EX Mask Quality at 256px — Resampling Mode Comparison")
    log(f"Checkpoint: {CKPT.name}  (run1, efficientnet-b2, threshold={THRESHOLD})")
    log(f"Target resolution: {TARGET_HW[0]}x{TARGET_HW[1]}")
    log(f"GT always downsampled with maxpool; prediction tested across {MODES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    ckpt  = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    log(f"Model loaded: epoch={ckpt.get('epoch')}  recall={ckpt.get('recall', 0):.4f}")

    for split, folder in [("Training Set", "a. Training Set"),
                          ("Testing Set",  "b. Testing Set")]:
        t0 = time.time()
        eval_split(split, folder, folder, model, device)
        log(f"\n  ({(time.time()-t0)/60:.1f} min)")

    log("\nDone.")
    _log.close()

    if args.drive_dir:
        import subprocess as _sp
        r = _sp.run(["rclone", "copy", str(LOG), args.drive_dir], capture_output=True, text=True)
        print("Drive backup:", "OK" if r.returncode == 0 else r.stderr.strip())
