"""
scripts/run_eval.py
Full evaluation harness on a holdout set.
Computes privacy / utility / realism metrics and generates scorecard.

Usage:
    python scripts/run_eval.py \
        --original  /data/eyepacs/holdout/ \
        --deid      /data/deid_output/ \
        --output    /data/evals/scorecards/ \
        --retfound  /workspace/RETFound \
        --weights   /workspace/models/RETFound_mae_natureCFP.pth \
        --model-name "AttentionUNet_ModelA + LaMa" \
        --device    cuda
"""

import argparse
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval import reid, realism, scorecard


def parse_args():
    p = argparse.ArgumentParser(description='RealizeMD eval harness')
    p.add_argument('--original',   required=True, help='Folder of original fundus images')
    p.add_argument('--deid',       required=True, help='Folder of de-identified images')
    p.add_argument('--output',     required=True, help='Where to save scorecards')
    p.add_argument('--retfound',   default='/workspace/RETFound', help='RETFound repo dir')
    p.add_argument('--weights',    required=True, help='RETFound weights .pth')
    p.add_argument('--model-name', default='Model A + LaMa')
    p.add_argument('--config',     default='configs/default.yaml')
    p.add_argument('--device',     default='cuda')
    p.add_argument('--n', '--limit', type=int, default=None, dest='n', help='Limit to N images')
    return p.parse_args()


def load_images(folder: Path, suffix: str = '', n: int = None, stems_to_keep: set = None) -> dict:
    """Load {stem: img_rgb} from a folder. suffix filters by name ending."""
    paths = sorted(
        list(folder.rglob('*.png')) +
        list(folder.rglob('*.jpeg')) +
        list(folder.rglob('*.jpg'))
    )
    if suffix:
        paths = [p for p in paths if p.stem.endswith(suffix)]
    if stems_to_keep is not None:
        paths = [p for p in paths if p.stem.replace(suffix, '') in stems_to_keep]
    if n:
        paths = paths[:n]

    images = {}
    for p in tqdm(paths, desc=f'Loading {folder.name}'):
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.resize(img, (512, 512))
            images[p.stem.replace(suffix, '')] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return images


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get('eval', {})

    print('\n=== Loading images ===')
    # Auto-detect suffix (e.g., _deid, _deid_dil15, _deid_dil40, etc.)
    deid_dir = Path(args.deid)
    candidate_paths = sorted(
        list(deid_dir.rglob('*.png')) +
        list(deid_dir.rglob('*.jpeg')) +
        list(deid_dir.rglob('*.jpg'))
    )
    suffix = '_deid'  # fallback default
    for p in candidate_paths:
        if '_deid' in p.stem:
            idx = p.stem.find('_deid')
            suffix = p.stem[idx:]
            break
    print(f"🔧 Detected de-identified image suffix: '{suffix}'")

    # Load deid images first to filter originals
    deid_images = load_images(deid_dir, suffix=suffix, n=args.n)
    deid_stems = set(deid_images.keys())
    
    original_images = load_images(Path(args.original), stems_to_keep=deid_stems)

    # Match stems
    common = set(original_images) & set(deid_images)
    print(f'Matched {len(common)} image pairs')
    original_images = {k: v for k, v in original_images.items() if k in common}
    deid_images     = {k: v for k, v in deid_images.items() if k in common}

    # ── Privacy metric ────────────────────────────────────────────────────────
    print('\n=== Privacy: RETFound re-id rate ===')
    reid.load_retfound(args.weights, args.retfound, device=args.device)

    print('Embedding originals...')
    orig_embeddings = reid.embed_batch(original_images)
    print('Embedding de-identified...')
    deid_embeddings = reid.embed_batch(deid_images)

    reid_results = reid.compute_reid_rate(orig_embeddings, deid_embeddings)
    print(f'  Rank-1 re-id rate: {reid_results["rank1_rate"]:.4f}')
    print(f'  vs. random:        {reid_results["ratio_vs_random"]:.2f}×')
    print(f'  Same-patient AUC:  {reid_results["same_patient_auc"]:.4f}')
    print(f'  Result: {"✅ PASS" if reid_results["pass"] else "❌ FAIL"}')

    # ── Utility metric ────────────────────────────────────────────────────────
    # Placeholder — DR classifier eval requires EyePACS labels
    # Victoria's task per the supervisor's plan
    utility_results = {
        'dr_auc_original':      None,
        'dr_auc_deid':          None,
        'preservation_ratio':   None,
        'dr_auc_deid_low_grade': None,
        'dr_auc_deid_high_grade': None,
        'pass':                 False,
        'note':                 'Pending DR classifier finetuning (Victoria)',
    }
    print('\n=== Utility: DR AUC ===')
    print('  ⏳ Pending DR classifier — Victoria\'s task')

    # ── Realism metrics ───────────────────────────────────────────────────────
    print('\n=== Realism: SSIM / LPIPS / FID ===')
    realism_results = realism.compute_ssim_lpips(original_images, deid_images)
    print(f'  SSIM mean:  {realism_results["ssim_mean"]:.4f}  (target 0.65–0.85)')

    fid_score = realism.compute_fid(
        list(original_images.values()),
        list(deid_images.values()),
        device=args.device,
    )
    realism_results['fid'] = fid_score
    realism_results['overall_pass'] = (
        realism_results.get('ssim_pass', False) and
        (fid_score is not None and fid_score < 30)
    )
    if fid_score:
        print(f'  FID:        {fid_score:.1f}  (target < 30)')

    # ── Scorecard ─────────────────────────────────────────────────────────────
    print('\n=== Generating scorecard ===')
    card = scorecard.generate(
        model_name=args.model_name,
        reid_results=reid_results,
        utility_results=utility_results,
        realism_results=realism_results,
        n_patients=len(common),
        n_images=len(common),
        output_dir=args.output,
    )
    print(card)


if __name__ == '__main__':
    main()
