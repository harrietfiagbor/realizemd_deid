"""
generate_controlnet_masks.py

Task 5: generate the multi-channel ControlNet training masks Adam asked for --
Model A vessel + EX/HE lesion masks, as a 3-channel control image per training
image, at 512px (the ControlNet training resolution).

Control image channel mapping (RGB):  R = vessel,  G = EX,  B = HE.

Per-source mask provenance (--mode):
  ddr      : vessel = Model A; EX/HE = the trained detectors (ex_run1 / he_run5).
  idrid    : vessel = Model A; EX/HE = IDRiD ground-truth .tif (higher quality
             than detector, "where available" per Adam). Pass --idrid-gt-root.
  healthy  : vessel = Model A; EX/HE = EMPTY (all-zero) -- teaches "healthy = clean".

Everything is done at 512px: images resized to 512 first, so each detector sees a
single native 512 patch (patch_size the detectors were trained with) and the masks
land at exactly the resolution the generator sees. (If sample QC shows EX/HE too
sparse at 512, switch to full-res sliding-window detection -- flagged, not assumed.)

CROSS-VALIDATION built in (per the always-cross-validate rule):
  - Asserts the EX/HE checkpoint identity (epoch/recall metadata) matches the
    validated ex_run1 / he_run5 before generating anything; aborts on mismatch.
  - Logs, per image, vessel/EX/HE pixel counts and a status (ok / load_fail /
    gt_missing), so a silent processing failure is never indistinguishable from a
    genuinely empty (healthy) mask.
  - Writes a per-image CSV summary alongside the masks.
  - --limit N  : process only the first N images (for the sample QC run before the
    full batch).

RunPod usage (sample first, then full):
  python scripts/generate_controlnet_masks.py \
      --src /workspace/data/ddr_1135 --mode ddr \
      --vessel-weights "/workspace/arkan_unet/Trained models/retina_attentionUnet_150epochs.hdf5" \
      --ex-ckpt /workspace/models/ex_detector/ex_detector_best.pth \
      --he-ckpt /workspace/models/he_detector/he_detector_best.pth \
      --out /workspace/diffusion_retina_victoria/masks_ddr --limit 20
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> pipeline importable
from pipeline import segmentation

import cv2
import numpy as np
from pathlib import Path

SIZE = 512

# validated checkpoint fingerprints (metadata), from decision.md / this session's verification
EXPECTED = {
    "ex": {"epoch": 5,  "recall_round": 0.764},   # ex_run1, seed 42
    "he": {"epoch": 35, "recall_round": 0.708},   # he_run5, seed 42
}


def preprocess_for_detector(img_bgr):
    """CLAHE on LAB L-channel -> RGB. Exactly as the detectors were trained/evaluated."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def load_detector(ckpt_path, device, which):
    import torch
    import segmentation_models_pytorch as smp
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # cross-validate identity before using it
    exp = EXPECTED[which]
    ep, rec = ckpt.get("epoch"), ckpt.get("recall")
    if ep != exp["epoch"] or rec is None or round(rec, 3) != exp["recall_round"]:
        raise SystemExit(
            f"[ABORT] {which.upper()} checkpoint identity mismatch: got epoch={ep} recall={rec}, "
            f"expected epoch={exp['epoch']} recall~{exp['recall_round']}. "
            f"This is not the validated {which}_run checkpoint -- fix before generating masks.")
    model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    thr = ckpt.get("threshold", 0.35)
    print(f"  {which.upper()} detector OK: epoch={ep} recall={rec:.4f} threshold={thr}")
    return model, thr


def detect_512(model, img_bgr_512, device, threshold):
    """Single native-512 patch inference -> binary mask (0/255)."""
    import torch
    pre = preprocess_for_detector(img_bgr_512)                 # 512x512x3 RGB
    t = torch.from_numpy(pre.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
    return (prob > threshold).astype(np.uint8) * 255


def vessel_512(img_rgb_512, threshold=0.5):
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_rgb_512[:, :, 1])
    mask = segmentation.predict({"green_clahe": g, "enhanced_rgb": img_rgb_512}, threshold=threshold)
    return (mask > 0).astype(np.uint8) * 255


def idrid_gt_512(gt_root, stem, lesion):
    """Load an IDRiD GT .tif (lesion in {'EX','HE'}) for `stem` from either split, resize to 512."""
    gt_root = Path(gt_root)
    sub = "3. Hard Exudates" if lesion == "EX" else "2. Haemorrhages"
    for split in ["a. Training Set", "b. Testing Set"]:
        for ext in ["tif", "tiff", "png"]:
            p = gt_root / split / sub / f"{stem}_{lesion}.{ext}"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                return (m > 0).astype(np.uint8) * 255
    return None  # not all IDRiD images have both lesion types


def main():
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of source images (.jpg/.png/.jpeg)")
    ap.add_argument("--mode", required=True, choices=["ddr", "idrid", "healthy"])
    ap.add_argument("--vessel-weights", required=True)
    ap.add_argument("--ex-ckpt", default=None, help="required for mode=ddr")
    ap.add_argument("--he-ckpt", default=None, help="required for mode=ddr")
    ap.add_argument("--idrid-gt-root", default=None, help="required for mode=idrid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="process only first N (sample QC run)")
    args = ap.parse_args()

    device = "cuda"
    import torch
    if not torch.cuda.is_available():
        device = "cpu"; print("WARNING: CUDA not available, running on CPU (slow).")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "control").mkdir(exist_ok=True)

    print("Loading Model A (vessel)...")
    segmentation.load_model(args.vessel_weights)

    ex_model = he_model = None; ex_thr = he_thr = 0.35
    if args.mode == "ddr":
        assert args.ex_ckpt and args.he_ckpt, "mode=ddr needs --ex-ckpt and --he-ckpt"
        print("Loading EX detector..."); ex_model, ex_thr = load_detector(args.ex_ckpt, device, "ex")
        print("Loading HE detector..."); he_model, he_thr = load_detector(args.he_ckpt, device, "he")
    if args.mode == "idrid":
        assert args.idrid_gt_root, "mode=idrid needs --idrid-gt-root"

    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif")
    files = sorted({p for e in exts for p in Path(args.src).glob(e)})
    if args.limit:
        files = files[:args.limit]
    print(f"\n{len(files)} images, mode={args.mode}\n")

    rows = []
    n_load_fail = n_gt_missing = n_ex_empty = n_he_empty = 0
    for i, fp in enumerate(files):
        stem = fp.stem
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            n_load_fail += 1
            rows.append(dict(name=stem, status="load_fail", vessel_px=0, ex_px=0, he_px=0))
            print(f"[{i+1}/{len(files)}] {stem}  LOAD_FAIL")
            continue

        img_bgr_512 = cv2.resize(img_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        img_rgb_512 = cv2.cvtColor(img_bgr_512, cv2.COLOR_BGR2RGB)

        v = vessel_512(img_rgb_512)

        status = "ok"
        if args.mode == "ddr":
            ex = detect_512(ex_model, img_bgr_512, device, ex_thr)
            he = detect_512(he_model, img_bgr_512, device, he_thr)
        elif args.mode == "idrid":
            ex = idrid_gt_512(args.idrid_gt_root, stem, "EX")
            he = idrid_gt_512(args.idrid_gt_root, stem, "HE")
            if ex is None and he is None:
                status = "gt_missing"; n_gt_missing += 1
            ex = ex if ex is not None else np.zeros((SIZE, SIZE), np.uint8)
            he = he if he is not None else np.zeros((SIZE, SIZE), np.uint8)
        else:  # healthy
            ex = np.zeros((SIZE, SIZE), np.uint8)
            he = np.zeros((SIZE, SIZE), np.uint8)

        if int(ex.sum()) == 0: n_ex_empty += 1
        if int(he.sum()) == 0: n_he_empty += 1

        control = np.stack([v, ex, he], axis=-1)  # R=vessel, G=EX, B=HE
        cv2.imwrite(str(out / "control" / f"{stem}_control.png"),
                    cv2.cvtColor(control, cv2.COLOR_RGB2BGR))

        rows.append(dict(name=stem, status=status,
                         vessel_px=int((v > 0).sum()), ex_px=int((ex > 0).sum()), he_px=int((he > 0).sum())))
        if (i + 1) % 50 == 0 or i < 5:
            print(f"[{i+1}/{len(files)}] {stem}  v={rows[-1]['vessel_px']} ex={rows[-1]['ex_px']} he={rows[-1]['he_px']} {status}")

    with open(out / "mask_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "vessel_px", "ex_px", "he_px"])
        w.writeheader(); w.writerows(rows)

    n = len(files)
    print("\n" + "=" * 60)
    print(f"mode={args.mode}  processed={n}")
    print(f"  load_fail={n_load_fail}  gt_missing={n_gt_missing}")
    print(f"  EX-empty masks: {n_ex_empty}/{n}   HE-empty masks: {n_he_empty}/{n}")
    print(f"  (empty is EXPECTED for healthy / low-grade; load_fail is a REAL problem)")
    print(f"control images -> {out/'control'}/   summary -> {out/'mask_summary.csv'}")


if __name__ == "__main__":
    main()
