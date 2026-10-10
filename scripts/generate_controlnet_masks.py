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

Output control images are 512px. Vessels: Model A on the 512-resized image (its fixed
input). EX/HE detectors: full-res sliding window at each detector's own trained window
(EX 512/384, HE 768/640 -- decision.md L852), mask then downsampled to 512. Shrinking the
whole image to 512 for the detectors (old detect_512 path) collapses recall -- do not.

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


# Each detector's VALIDATED inference window (decision.md L852): EX was trained/evaluated on
# 512px patches (stride 384), HE on 768px patches (stride 640). Running either on a whole image
# shrunk to 512 (detect_512) puts lesions far below training scale and collapses recall --
# that mistake produced the unusable grade3plus masks. Do not use detect_512 for real masks.
DETECTOR_WINDOW = {"ex": (512, 384), "he": (768, 640)}


def load_od_detector(ckpt_path, device):
    """Learned optic-disc detector od_run1_b0 (decision.md L947-985). Identity asserted."""
    import torch
    import segmentation_models_pytorch as smp
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    if c.get("encoder") != "efficientnet-b0" or c.get("epoch") != 5 or round(c.get("dice", 0), 3) != 0.963:
        raise SystemExit(f"[ABORT] OD checkpoint is not od_run1_b0: encoder={c.get('encoder')} "
                         f"epoch={c.get('epoch')} dice={c.get('dice')}")
    m = smp.Unet(encoder_name=c["encoder"], in_channels=3, classes=1)
    m.load_state_dict(c["state_dict"]); m.to(device).eval()
    print(f"  OD detector OK: {c['encoder']} epoch={c['epoch']} dice={c['dice']:.4f} img_size={c['img_size']}")
    return m, c["img_size"]


def od_mask_fullres(od_model, od_size, img_bgr_full, device):
    """Exactly check_od_contamination_learned.get_od_mask (branch victoria-ex-od-exclusion-wiring):
    CLAHE-preprocessed image resized whole to img_size, sigmoid>0.5, NEAREST back to full res."""
    import torch
    rgb = preprocess_for_detector(img_bgr_full)
    H, W = rgb.shape[:2]
    r = cv2.resize(rgb, (od_size, od_size))
    with torch.no_grad():
        t = torch.from_numpy(r.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        prob = torch.sigmoid(od_model(t.to(device)))[0, 0].cpu().numpy()
    return cv2.resize((prob > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0


def detect_fullres_512(model, img_bgr_full, device, threshold, which, exclude_full=None):
    """Full-res sliding window at the detector's own trained patch/stride, then an any-pixel
    downsample of the binary mask to 512 so small lesions are not lost. `exclude_full` (bool,
    full-res) is removed BEFORE downsampling -- used for EX optic-disc exclusion (decision L1087)."""
    pred = detect_fullres(model, img_bgr_full, device, threshold, which, exclude_full)
    mask = pred.astype(np.uint8) * 255
    return (cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255


def detect_fullres(model, img_bgr_full, device, threshold, which, exclude_full=None):
    """Full-res boolean prediction (before any downsampling)."""
    import torch
    patch, stride = DETECTOR_WINDOW[which]
    rgb = preprocess_for_detector(img_bgr_full)
    H, W = rgb.shape[:2]
    prob = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = sorted(set(list(range(0, max(H - patch, 1), stride)) + [max(H - patch, 0)]))
    xs = sorted(set(list(range(0, max(W - patch, 1), stride)) + [max(W - patch, 0)]))
    with torch.no_grad():
        for y in ys:
            for x in xs:
                c = rgb[y:y + patch, x:x + patch]
                t = torch.from_numpy(c.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                prob[y:y + patch, x:x + patch] += torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
                cnt[y:y + patch, x:x + patch] += 1
    pred = (prob / np.maximum(cnt, 1)) > threshold
    if exclude_full is not None:
        pred &= ~exclude_full
    return pred


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


def gt_from_dir(gt_dir, stem):
    """Load a GT mask for `stem` from a flat label dir (DDR lesion_segmentation
    layout: label/EX/<stem>.tif|png). Returns 512 binary mask or None."""
    if gt_dir is None:
        return None
    gt_dir = Path(gt_dir)
    for ext in ["tif", "tiff", "png", "jpg"]:
        p = gt_dir / f"{stem}.{ext}"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            return (m > 0).astype(np.uint8) * 255
    return None


def main():
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of source images (.jpg/.png/.jpeg)")
    ap.add_argument("--mode", required=True, choices=["ddr", "idrid", "healthy", "gt"],
                    help="ddr=detector masks; idrid=IDRiD GT tree; gt=GT from --ex-gt-dir/--he-gt-dir "
                         "(DDR lesion_segmentation: label/EX, label/HE); healthy=empty lesion channels")
    ap.add_argument("--vessel-weights", required=True)
    ap.add_argument("--ex-ckpt", default=None, help="required for mode=ddr")
    ap.add_argument("--he-ckpt", default=None, help="required for mode=ddr")
    ap.add_argument("--od-ckpt", default=None,
                    help="mode=ddr: learned OD checkpoint (od_run1_b0) for EX disc exclusion (decision L1087)")
    ap.add_argument("--idrid-gt-root", default=None, help="required for mode=idrid")
    ap.add_argument("--ex-gt-dir", default=None, help="mode=gt: flat dir of EX GT masks named <stem>.tif/png")
    ap.add_argument("--he-gt-dir", default=None, help="mode=gt: flat dir of HE GT masks named <stem>.tif/png")
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
        assert args.od_ckpt, "mode=ddr needs --od-ckpt (EX disc exclusion is the decided production setting)"
        print("Loading OD detector..."); od_model, od_size = load_od_detector(args.od_ckpt, device)
    if args.mode == "idrid":
        assert args.idrid_gt_root, "mode=idrid needs --idrid-gt-root"
    if args.mode == "gt":
        assert args.ex_gt_dir or args.he_gt_dir, (
            "mode=gt needs at least one of --ex-gt-dir / --he-gt-dir -- "
            "without them every lesion channel would be silently empty")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif")
    files = sorted({p for e in exts for p in Path(args.src).glob(e)})
    if args.limit:
        files = files[:args.limit]
    print(f"\n{len(files)} images, mode={args.mode}\n")

    rows = []
    n_load_fail = n_gt_missing = n_ex_empty = n_he_empty = n_skip = 0
    for i, fp in enumerate(files):
        stem = fp.stem
        if (out / "control" / f"{stem}_control.png").exists():   # resume: skip already-done
            n_skip += 1
            continue
        print(f"[{i+1}/{len(files)}] {stem}", flush=True)
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
            disc = od_mask_fullres(od_model, od_size, img_bgr, device)
            ex = detect_fullres_512(ex_model, img_bgr, device, ex_thr, "ex", exclude_full=disc)
            he = detect_fullres_512(he_model, img_bgr, device, he_thr, "he")
        elif args.mode == "idrid":
            ex = idrid_gt_512(args.idrid_gt_root, stem, "EX")
            he = idrid_gt_512(args.idrid_gt_root, stem, "HE")
            if ex is None and he is None:
                status = "gt_missing"; n_gt_missing += 1
            ex = ex if ex is not None else np.zeros((SIZE, SIZE), np.uint8)
            he = he if he is not None else np.zeros((SIZE, SIZE), np.uint8)
        elif args.mode == "gt":   # real GT masks from flat label dirs (DDR lesion_segmentation)
            ex = gt_from_dir(args.ex_gt_dir, stem)
            he = gt_from_dir(args.he_gt_dir, stem)
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
    n_done = len(list((out / "control").glob("*_control.png")))
    print("\n" + "=" * 60)
    print(f"mode={args.mode}  total={n}  skipped(already done)={n_skip}  newly processed={n - n_skip}")
    print(f"  control masks on disk now: {n_done}/{n}")
    print(f"  this run: load_fail={n_load_fail}  gt_missing={n_gt_missing}")
    print(f"  this run empty masks: EX={n_ex_empty} HE={n_he_empty} (empty EXPECTED for low-grade; load_fail is REAL)")
    print(f"control images -> {out/'control'}/   summary(this run) -> {out/'mask_summary.csv'}")

    # loud guard: in GT modes, a high gt_missing fraction means the stem-matching/dir is WRONG,
    # which would silently produce all-empty lesion channels. Fail loud, don't let it pass as "healthy".
    processed = n - n_skip
    if args.mode in ("gt", "idrid") and processed > 0 and n_gt_missing > 0.2 * processed:
        print("\n" + "!" * 60)
        print(f"WARNING: {n_gt_missing}/{processed} images had NO matching GT mask "
              f"({100*n_gt_missing/processed:.0f}%).")
        print("This almost certainly means the image stems don't match the GT filenames,")
        print("or the --ex-gt-dir/--he-gt-dir/--idrid-gt-root path is wrong. The lesion")
        print("channels for those images are EMPTY -- do NOT use these as training data")
        print("until the structure is confirmed. Check a few stems vs the label dir contents.")
        print("!" * 60)


if __name__ == "__main__":
    main()
