#!/usr/bin/env python3
"""
select_triptych_candidates_stage2.py

Stage 2 (GPU, pod) of triptych candidate selection: runs the validated
learned EX and HE detectors on Stage 1's 300-image shortlist, ranks by
lesion strength, and picks the final ~15-20 candidates for Harriet's
inversion-fidelity gate (Adam's Step 1).

Rationale for ranking: Adam wants "real, unambiguous DR findings" -- a
demo image with only a couple faint HE dots isn't a strong test of lesion
survival through inversion. Ranks candidates by a combined score that
requires BOTH lesion types to be reasonably present (not just one type
dominating), since a real go/no-go check on inversion fidelity is more
convincing across both hard exudates and haemorrhages together.

Score = min(ex_blob_count, he_blob_count) weighted by total area -- a
deliberately conservative combination so an image with lots of EX but zero
HE doesn't rank above a balanced case.

Usage (pod):
    python realizemd_deid/scripts/select_triptych_candidates_stage2.py \
        --pool_dir /workspace/triptych/stage1_pool \
        --ex_ckpt /workspace/models/ex_detector/ex_detector_best.pth \
        --he_ckpt /workspace/models/he_detector/he_detector_best.pth \
        --out_dir /workspace/triptych
"""
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from skimage import measure as sk_measure


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


def blob_stats(pred_bin):
    labeled = sk_measure.label(pred_bin)
    regions = sk_measure.regionprops(labeled)
    n = len(regions)
    area = int(pred_bin.sum())
    return n, area


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ex_ckpt = torch.load(args.ex_ckpt, map_location="cpu", weights_only=False)
    ex_model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1).to(device)
    ex_model.load_state_dict(ex_ckpt["state_dict"])
    ex_model.eval()
    print(f"EX loaded: epoch={ex_ckpt.get('epoch')} recall={ex_ckpt.get('recall', 0):.4f}")

    he_ckpt = torch.load(args.he_ckpt, map_location="cpu", weights_only=False)
    he_model = smp.Unet(encoder_name="efficientnet-b2", in_channels=3, classes=1).to(device)
    he_model.load_state_dict(he_ckpt["state_dict"])
    he_model.eval()
    print(f"HE loaded: epoch={he_ckpt.get('epoch')} recall={he_ckpt.get('recall', 0):.4f}")

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    with open(pool_dir / "triptych_candidates_stage1.csv") as f:
        next(f)
        for line in f:
            stem = line.strip().split(",")[0]
            stems.append(stem)

    results = []
    for i, stem in enumerate(stems):
        img_path = pool_dir / f"{stem}.jpeg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = preprocess(img_bgr)

        ex_prob = get_prob_map(ex_model, img_rgb, device, 512, 384)
        ex_bin = (ex_prob > 0.35).astype(np.uint8)
        ex_n, ex_area = blob_stats(ex_bin)

        he_prob = get_prob_map(he_model, img_rgb, device, 768, 640)
        he_bin = (he_prob > 0.35).astype(np.uint8)
        he_n, he_area = blob_stats(he_bin)

        score = min(ex_n, he_n) * np.log1p(ex_area + he_area)
        results.append((stem, ex_n, ex_area, he_n, he_area, score))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(stems)} processed")

    results.sort(key=lambda r: r[-1], reverse=True)

    out_csv = out_dir / "triptych_candidates_stage2_ranked.csv"
    with open(out_csv, "w") as f:
        f.write("stem,ex_blob_count,ex_area_px,he_blob_count,he_area_px,score\n")
        for r in results:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nSaved -> {out_csv}")
    print("\nTop 20 candidates:")
    for r in results[:20]:
        print(f"  {r[0]:14s} ex_blobs={r[1]:3d} ex_area={r[2]:6d}  he_blobs={r[3]:3d} he_area={r[4]:6d}  score={r[5]:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool_dir", required=True)
    p.add_argument("--ex_ckpt", required=True)
    p.add_argument("--he_ckpt", required=True)
    p.add_argument("--out_dir", default="triptych")
    main(p.parse_args())
