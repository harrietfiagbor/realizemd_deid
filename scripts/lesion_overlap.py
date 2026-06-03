"""
scripts/lesion_overlap.py
─────────────────────────────────────────────────────────────────────────────
Lesion-mask overlap audit for IDRiD grade 2–4 images.

Answers the question Adam posed:
  "Of all GT lesion area, what fraction ends up inside the final inpaint mask?"
  Stratified by EX / HE / MA.

If HE pixel-preservation (fraction of HE pixels OUTSIDE the inpaint mask) ≥ 0.98
→ raw 0.22 HE recall is irrelevant; proceed to Path A.
If HE preservation < 0.95
→ real destruction; learned IDRiD detector becomes urgent before pilot.

IDRiD GT mask naming convention (adjust --gt-dir structure if yours differs):
  <gt_dir>/
    EX/<image_id>_EX.tif      (binary: 255 = lesion)
    HE/<image_id>_HE.tif
    MA/<image_id>_MA.tif

Usage:
    python scripts/lesion_overlap.py \\
        --images   /data/idrid/images/          \\  # fundus PNGs
        --gt-dir   /data/idrid/GT/              \\  # IDRiD GT folder
        --output   /data/evals/overlap/         \\  # where to save results CSV + summary
        --vessel-dilation  20                   \\  # px — MUST match default.yaml
        --lesion-dilation  3                    \\  # px — MUST match default.yaml
        --device   cuda

Interpretation gates (Adam's spec):
    HE preservation ≥ 0.98  →  ✅ proceed with Path A
    HE preservation ∈ [0.95, 0.98)  →  ⚠️  monitor; flag in scorecard
    HE preservation < 0.95  →  ❌ learned HE detector required before pilot
"""

import argparse
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import preprocessing, segmentation, masking
import yaml


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Lesion-inpaint-mask overlap audit')
    p.add_argument('--images',            required=True, help='Folder of IDRiD fundus images')
    p.add_argument('--gt-dir',            required=True, help='IDRiD GT root (EX/ HE/ MA/ subdirs)')
    p.add_argument('--output',            required=True, help='Where to save CSV + summary')
    p.add_argument('--config',            default='configs/default.yaml')
    p.add_argument('--vessel-dilation',   type=int, default=None,
                   help='Override vessel dilation kernel (px). Defaults to config value.')
    p.add_argument('--lesion-dilation',   type=int, default=None,
                   help='Override lesion-exclusion dilation (px). Defaults to config value.')
    p.add_argument('--device',            default='cuda')
    p.add_argument('--n',                 type=int, default=None, help='Limit to N images')
    return p.parse_args()


# ── GT loading ────────────────────────────────────────────────────────────────

LESION_TYPES = ['EX', 'HE', 'MA']


def load_gt_masks(gt_dir: Path, image_id: str, target_size: int = 512) -> dict:
    """
    Load IDRiD binary GT masks for one image.
    Returns {lesion_type: uint8 (H,W) mask} or None per type if not present.

    IDRiD TIF masks are already binary (pixel 255 = lesion).
    We resize with INTER_NEAREST to preserve binary values.
    """
    masks = {}
    for lt in LESION_TYPES:
        candidates = list((gt_dir / lt).glob(f'{image_id}_{lt}.*'))
        if not candidates:
            masks[lt] = None
            continue
        raw = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            masks[lt] = None
            continue
        resized = cv2.resize(raw, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        masks[lt] = (resized > 127).astype(np.uint8) * 255
    return masks


# ── Per-image overlap computation ─────────────────────────────────────────────

def compute_overlap_for_image(
    img_path: Path,
    gt_dir: Path,
    vessel_dilation: int,
    lesion_dilation: int,
    cfg: dict,
    device: str,
) -> dict | None:
    """
    For one IDRiD image:
      1. Run preprocessing + vessel segmentation (same as production pipeline)
      2. Build inpaint mask using the provided dilation params
      3. For each lesion type, compute:
           gt_px         = total GT lesion pixels
           inside_mask   = GT pixels that land inside the inpaint mask
           overlap_frac  = inside_mask / gt_px  (fraction at risk of being inpainted)
           preservation  = 1 - overlap_frac     (fraction safely outside)

    Returns a flat dict with per-type stats, or None on processing error.
    """
    image_id = img_path.stem
    target_size = cfg.get('pipeline', {}).get('target_size', 512)

    # ── Vessel segmentation ────────────────────────────────────────────────────
    try:
        preprocessed = preprocessing.preprocess(
            img_path,
            target_size=target_size,
            clahe_clip=cfg.get('preprocessing', {}).get('clahe_clip', 2.0),
            clahe_tile=tuple(cfg.get('preprocessing', {}).get('clahe_tile', [8, 8])),
        )
        vessel_mask = segmentation.predict(
            preprocessed,
            threshold=cfg.get('segmentation', {}).get('threshold', 0.5),
        )
    except Exception as e:
        print(f'  ⚠️  Segmentation failed for {image_id}: {e}')
        return None

    # ── GT masks ──────────────────────────────────────────────────────────────
    gt_masks = load_gt_masks(gt_dir, image_id, target_size)

    # ── Build inpaint mask (vessels − lesions, dilated) ───────────────────────
    # For the overlap test we use an EMPTY detected-lesion mask.
    # This is intentional: we want to measure the worst-case overlap (before
    # any lesion exclusion). The lesion exclusion only helps if lesions are
    # detected; missed lesions get zero exclusion. This gives Adam the
    # "how much GT lesion is at risk if the detector completely misses it" number.
    empty_lesion_mask = np.zeros_like(vessel_mask)
    mask_result = masking.build_inpaint_mask(
        vessel_mask=vessel_mask,
        lesion_mask=empty_lesion_mask,
        vessel_dilation_kernel=vessel_dilation,
        lesion_dilation_kernel=lesion_dilation,
    )
    inpaint_mask = mask_result['inpaint_mask']  # uint8, 255 = region to be inpainted

    # ── Overlap stats per lesion type ─────────────────────────────────────────
    row = {
        'image_id':       image_id,
        'vessel_pct':     mask_result['stats']['vessel_pct'],
        'inpaint_pct':    mask_result['stats']['inpaint_pct'],
    }

    inpaint_binary = (inpaint_mask > 0)

    for lt in LESION_TYPES:
        gt = gt_masks.get(lt)
        if gt is None:
            row.update({
                f'{lt}_gt_px':       None,
                f'{lt}_inside_px':   None,
                f'{lt}_overlap_frac': None,
                f'{lt}_preservation': None,
            })
            continue

        gt_binary  = (gt > 0)
        gt_px      = int(gt_binary.sum())
        inside_px  = int((gt_binary & inpaint_binary).sum())

        overlap_frac  = inside_px / gt_px if gt_px > 0 else 0.0
        preservation  = 1.0 - overlap_frac

        row.update({
            f'{lt}_gt_px':        gt_px,
            f'{lt}_inside_px':    inside_px,
            f'{lt}_overlap_frac': round(overlap_frac, 4),
            f'{lt}_preservation': round(preservation, 4),
        })

    return row


# ── Interpretation gate ───────────────────────────────────────────────────────

GATE_PASS    = 0.98   # HE preservation ≥ 0.98 → proceed with Path A
GATE_WARN    = 0.95   # HE preservation ∈ [0.95, 0.98) → monitor + flag
# < 0.95 → learned detector required before pilot


def interpret(mean_preservation: float, lesion_type: str) -> str:
    if lesion_type != 'HE':
        return '(no hard gate)'
    if mean_preservation >= GATE_PASS:
        return f'✅ PASS (≥{GATE_PASS}) — proceed with Path A'
    elif mean_preservation >= GATE_WARN:
        return f'⚠️  WARN (≥{GATE_WARN}, <{GATE_PASS}) — flag in scorecard'
    else:
        return f'❌ FAIL (<{GATE_WARN}) — learned HE detector required before pilot'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vessel_dilation = args.vessel_dilation or cfg.get('vessel_mask', {}).get('dilation_kernel', 20)
    lesion_dilation = args.lesion_dilation or cfg.get('pathology', {}).get('lesion_dilation_kernel', 3)

    print(f'\n🔧 Dilation params:')
    print(f'   vessel_dilation_kernel = {vessel_dilation}px')
    print(f'   lesion_dilation_kernel = {lesion_dilation}px  (used for exclusion zone around detected lesions)')
    print(f'   NOTE: overlap is computed with NO detected lesions — worst-case / missed-detection scenario\n')

    # Load segmentation model once
    segmentation.load_model(
        cfg.get('segmentation', {}).get('weights', 'models/attention_unet/retina_attentionUnet_150epochs.hdf5'),
        device=args.device,
    )

    image_paths = sorted(
        list(Path(args.images).rglob('*.png')) +
        list(Path(args.images).rglob('*.jpg')) +
        list(Path(args.images).rglob('*.jpeg'))
    )
    image_paths = [p for p in image_paths if not any(part.startswith('.') for part in p.parts)]
    if args.n:
        image_paths = image_paths[:args.n]

    print(f'Processing {len(image_paths)} images...\n')

    rows = []
    for path in tqdm(image_paths, desc='Overlap audit'):
        row = compute_overlap_for_image(
            img_path=path,
            gt_dir=Path(args.gt_dir),
            vessel_dilation=vessel_dilation,
            lesion_dilation=lesion_dilation,
            cfg=cfg,
            device=args.device,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        print('No results — check image paths and GT directory structure.')
        return

    df = pd.DataFrame(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*65)
    print('LESION–INPAINT MASK OVERLAP SUMMARY')
    print(f'Vessel dilation: {vessel_dilation}px  |  N images: {len(df)}')
    print('='*65)
    print(f'{"":4}  {"Mean GT px":>12}  {"Mean inside":>12}  {"Overlap%":>10}  {"Preservation":>13}  {"Gate"}')
    print('-'*65)

    for lt in LESION_TYPES:
        col_pres = f'{lt}_preservation'
        col_over = f'{lt}_overlap_frac'
        col_gt   = f'{lt}_gt_px'
        col_in   = f'{lt}_inside_px'

        valid = df[df[col_pres].notna()]
        if valid.empty:
            print(f'{lt:4}  {"N/A":>12}')
            continue

        mean_gt      = valid[col_gt].mean()
        mean_inside  = valid[col_in].mean()
        mean_overlap = valid[col_over].mean()
        mean_pres    = valid[col_pres].mean()

        gate = interpret(mean_pres, lt)
        print(f'{lt:4}  {mean_gt:>12.0f}  {mean_inside:>12.0f}  {mean_overlap*100:>9.2f}%  {mean_pres:>12.4f}    {gate}')

    print('='*65)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f'lesion_overlap_dil{vessel_dilation}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nPer-image results saved: {csv_path}')

    # Summary JSON
    import json
    summary = {
        'vessel_dilation_kernel': vessel_dilation,
        'lesion_dilation_kernel': lesion_dilation,
        'n_images': len(df),
        'note': 'Overlap computed with empty detected-lesion mask (worst-case / missed-detection scenario)',
    }
    for lt in LESION_TYPES:
        col_pres = f'{lt}_preservation'
        col_over = f'{lt}_overlap_frac'
        valid = df[df[col_pres].notna()]
        if valid.empty:
            continue
        summary[lt] = {
            'mean_overlap_frac': round(float(valid[col_over].mean()), 4),
            'mean_preservation': round(float(valid[col_pres].mean()), 4),
            'gate': interpret(float(valid[col_pres].mean()), lt),
        }

    summary_path = output_dir / f'lesion_overlap_summary_dil{vessel_dilation}.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Summary JSON saved:   {summary_path}')


if __name__ == '__main__':
    main()
