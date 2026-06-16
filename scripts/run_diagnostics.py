"""
scripts/run_diagnostics.py
Three targeted diagnostics for the RealizeMD de-identification pipeline.

Adam's brief (2026-06-08):
  Diag 1 — LoRA load verification
  Diag 2 — No-inpaint privacy baseline
  Diag 3 — FID curve across conditioning scales

Usage:
    # Run all three
    python scripts/run_diagnostics.py --all \
        --input  /workspace/data/eyepacs/images/ \
        --output /workspace/data/diagnostics/ \
        --weights models/attention_unet/retina_attentionUnet_150epochs.hdf5 \
        --device cuda

    # Run individually
    python scripts/run_diagnostics.py --diag lora  ...
    python scripts/run_diagnostics.py --diag baseline ...
    python scripts/run_diagnostics.py --diag fid_curve ...

    # Diag 1 needs fewer images (5 fixed-seed pairs)
    # Diag 2 + 3 share the same 10-image preprocessed batch

Outputs (all under --output dir):
    diag1_lora/        — side-by-side PNGs (lora_on vs lora_off), diff image,
                         diag1_lora_results.json
    diag2_baseline/    — diag2_baseline_results.json  (AUC table + narrative)
    diag3_fid_curve/   — per-scale output images + diag3_fid_curve_results.json
    diagnostics_summary.txt  — one-page brief for Adam
"""

import argparse
import json
import sys
import yaml
import copy
import cv2
import numpy as np
from pathlib import Path
from datetime import date
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import preprocessing, segmentation, pathology, masking, inpainting
from eval import reid, realism


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RealizeMD diagnostic suite')
    p.add_argument('--input',   required=True, help='Folder of fundus images')
    p.add_argument('--output',  required=True, help='Root output folder for all diagnostics')
    p.add_argument('--weights', required=True, help='Path to Model A .h5 weights')
    p.add_argument('--config',  default='configs/default.yaml')
    p.add_argument('--device',  default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--retfound-weights', default=None,
                   help='Path to RETFound weights (.pth). Required for Diag 2 & 3.')
    p.add_argument('--retfound-dir', default='/workspace/RETFound_MAE',
                   help='Path to RETFound repo (contains models_vit.py)')

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all',       action='store_true', help='Run all three diagnostics')
    mode.add_argument('--diag',      choices=['lora', 'baseline', 'fid_curve'],
                      help='Run a single diagnostic')

    p.add_argument('--n-lora',      type=int, default=5,
                   help='Images for Diag 1 LoRA comparison (default: 5)')
    p.add_argument('--n-images',    type=int, default=10,
                   help='Images for Diag 2 & 3 (default: 10)')
    p.add_argument('--lora-seed',   type=int, default=42,
                   help='Fixed seed for Diag 1 LoRA comparison')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):      return bool(obj)
        if isinstance(obj, np.integer):        return int(obj)
        if isinstance(obj, np.floating):       return float(obj)
        if isinstance(obj, np.ndarray):        return obj.tolist()
        return super().default(obj)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, cls=NumpyEncoder))


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_images(input_dir, n):
    paths = sorted(
        list(Path(input_dir).rglob('*.jpeg')) +
        list(Path(input_dir).rglob('*.jpg')) +
        list(Path(input_dir).rglob('*.png'))
    )
    if not paths:
        raise FileNotFoundError(f'No images found in {input_dir}')
    return paths[:n]


def preprocess_batch(image_paths, cfg):
    """Preprocess + segment + detect pathology for a list of images.
    Returns list of dicts ready for inpainting."""
    pp_cfg   = cfg.get('preprocessing', {})
    seg_cfg  = cfg.get('segmentation', {})
    mask_cfg = cfg.get('vessel_mask', {})
    path_cfg = cfg.get('pathology', {})
    dil      = mask_cfg.get('dilation_kernel', 5)

    reference_rgb = preprocessing.select_reference(image_paths)
    batch = []
    for img_path in tqdm(image_paths, desc='  Pre-processing'):
        try:
            preprocessed = preprocessing.preprocess(
                img_path,
                target_size=pp_cfg.get('target_size', 512),
                reference_rgb=reference_rgb,
            )
            vessel_mask = segmentation.predict(
                preprocessed,
                threshold=seg_cfg.get('threshold', 0.5)
            )
            lesion_result = pathology.detect_all(
                preprocessed['enhanced_rgb'], path_cfg
            )
            mask_result = masking.build_inpaint_mask(
                vessel_mask=vessel_mask,
                lesion_mask=lesion_result['combined'],
                vessel_dilation_kernel=dil,
            )
            batch.append({
                'path':        img_path,
                'stem':        img_path.stem,
                'preprocessed': preprocessed,
                'vessel_mask': vessel_mask,
                'mask_result': mask_result,
            })
        except Exception as e:
            print(f'  ⚠️  Pre-processing failed {img_path.name}: {e}')
    return batch


def run_inpaint_batch(batch, inp_cfg, device,
                      controlnet_conditioning_scale=None,
                      seed_override=None):
    """Run inpainting on a preprocessed batch.
    Returns {stem: deid_rgb} dict."""
    import torch
    outputs = {}
    for item in tqdm(batch, desc='  Inpainting'):
        try:
            seed = seed_override if seed_override is not None else \
                   int(torch.randint(0, 2**31, (1,)).item())
            deid = inpainting.inpaint(
                image_rgb=item['preprocessed']['original_rgb'],
                mask=item['mask_result']['inpaint_mask'],
                vessel_mask=item['vessel_mask'],
                device=device,
                seed=seed,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                fov=item['preprocessed']['fov'],
            )
            outputs[item['stem']] = deid
        except Exception as e:
            print(f'  ⚠️  Inpainting failed {item["path"].name}: {e}')
    return outputs


def originals_dict(batch):
    """Return {stem: original_rgb} from a preprocessed batch."""
    return {item['stem']: item['preprocessed']['original_rgb'] for item in batch}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 1 — LoRA Load Verification
# ─────────────────────────────────────────────────────────────────────────────

def diag1_lora(args, cfg, out_root):
    """
    Run 5 images with fixed seed, LoRA on vs LoRA off.
    Measures:
      - Per-pixel mean absolute difference (MAD)
      - Cosine similarity in pixel space (flattened)
    If LoRA is not applying, MAD ≈ 0 and cosine sim ≈ 1.0.
    """
    print('\n' + '='*60)
    print('DIAGNOSTIC 1 — LoRA Load Verification')
    print('='*60)

    out_dir = Path(out_root) / 'diag1_lora'
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(args.input, args.n_lora)
    print(f'  Images: {len(image_paths)}  |  Fixed seed: {args.lora_seed}')

    segmentation.load_model(args.weights)

    # ── Run WITH LoRA ─────────────────────────────────────────────────────────
    print('\n  [1/2] Running WITH LoRA ...')
    cfg_lora_on = copy.deepcopy(cfg)
    cfg_lora_on['inpainting']['device'] = args.device
    inpainting.load_model(cfg=cfg_lora_on.get('inpainting', {}), device=args.device)
    batch = preprocess_batch(image_paths, cfg_lora_on)
    outputs_on = run_inpaint_batch(batch, cfg_lora_on.get('inpainting', {}),
                                    args.device, seed_override=args.lora_seed)

    # ── Run WITHOUT LoRA ──────────────────────────────────────────────────────
    print('\n  [2/2] Running WITHOUT LoRA ...')
    cfg_lora_off = copy.deepcopy(cfg)
    cfg_lora_off['inpainting']['lora_weights'] = None
    cfg_lora_off['inpainting']['device'] = args.device

    # Need to reload pipeline with LoRA disabled — clear singleton first
    inpainting._sd_pipe = None
    inpainting.load_model(cfg=cfg_lora_off.get('inpainting', {}), device=args.device)
    outputs_off = run_inpaint_batch(batch, cfg_lora_off.get('inpainting', {}),
                                     args.device, seed_override=args.lora_seed)

    # ── Compare ───────────────────────────────────────────────────────────────
    results = []
    common_stems = [s for s in outputs_on if s in outputs_off]

    for stem in common_stems:
        img_on  = outputs_on[stem].astype(np.float32)
        img_off = outputs_off[stem].astype(np.float32)

        mad = float(np.mean(np.abs(img_on - img_off)))

        flat_on  = img_on.flatten()
        flat_off = img_off.flatten()
        cos_sim  = float(
            np.dot(flat_on, flat_off) /
            (np.linalg.norm(flat_on) * np.linalg.norm(flat_off) + 1e-8)
        )

        # Diff image (amplified ×5 for visibility)
        diff_amp = np.clip(np.abs(img_on - img_off) * 5, 0, 255).astype(np.uint8)

        # Save side-by-side: original | lora_on | lora_off | diff
        orig = next(
            item['preprocessed']['original_rgb']
            for item in batch if item['stem'] == stem
        )
        row = np.concatenate([
            orig,
            outputs_on[stem],
            outputs_off[stem],
            diff_amp,
        ], axis=1)
        cv2.imwrite(
            str(out_dir / f'{stem}_lora_compare.png'),
            cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
        )

        results.append({
            'stem':    stem,
            'mad':     round(mad, 4),
            'cos_sim': round(cos_sim, 6),
            'lora_active': mad > 2.0,  # heuristic: <2 MAD = effectively identical
        })
        print(f'    {stem}: MAD={mad:.2f}  cos_sim={cos_sim:.6f}  '
              f'{"✅ LoRA active" if mad > 2.0 else "❌ LoRA NOT applying"}')

    # ── Verdict ───────────────────────────────────────────────────────────────
    mean_mad = np.mean([r['mad'] for r in results])
    lora_confirmed = mean_mad > 2.0

    verdict = (
        f'Mean MAD across {len(results)} images: {mean_mad:.2f}\n'
        f'LoRA active: {"YES ✅" if lora_confirmed else "NO ❌ — outputs identical; LoRA is not loading/applying"}'
    )
    print(f'\n  {verdict}')

    summary = {
        'diagnostic': 'diag1_lora',
        'date': date.today().isoformat(),
        'n_images': len(results),
        'fixed_seed': args.lora_seed,
        'mean_mad': round(float(mean_mad), 4),
        'lora_confirmed': bool(lora_confirmed),
        'verdict': verdict,
        'per_image': results,
    }
    save_json(out_dir / 'diag1_lora_results.json', summary)
    print(f'  Saved → {out_dir}')
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 2 — No-Inpaint Privacy Baseline
# ─────────────────────────────────────────────────────────────────────────────

def diag2_baseline(args, cfg, out_root, batch=None):
    """
    Establish the actual de-identification effect of the pipeline.

    Three AUC measurements on the same image set:
      A. original → original  (upper bound; should be ~1.0)
      B. inpainted → original (what the pipeline actually achieves)
      C. AUC gap = A - B      (the actual de-id effect; we expect this to be tiny)

    Requires RETFound weights.
    """
    print('\n' + '='*60)
    print('DIAGNOSTIC 2 — No-Inpaint Privacy Baseline')
    print('='*60)

    if not args.retfound_weights:
        print('  ⚠️  --retfound-weights not provided. Skipping Diag 2.')
        return None

    out_dir = Path(out_root) / 'diag2_baseline'
    out_dir.mkdir(parents=True, exist_ok=True)

    if batch is None:
        image_paths = find_images(args.input, args.n_images)
        segmentation.load_model(args.weights)
        inpainting.load_model(cfg=cfg.get('inpainting', {}), device=args.device)
        batch = preprocess_batch(image_paths, cfg)

    originals = originals_dict(batch)

    # ── Load RETFound ─────────────────────────────────────────────────────────
    print('\n  Loading RETFound embedder ...')
    if args.retfound_dir and args.retfound_dir not in sys.path:
        sys.path.insert(0, args.retfound_dir)
    reid.load_retfound(
        weights_path=args.retfound_weights,
        retfound_dir=args.retfound_dir,
        device=args.device,
    )

    # ── A. Embed originals ────────────────────────────────────────────────────
    print('\n  [A] Embedding originals ...')
    orig_embeddings = reid.embed_batch(originals)

    # ── B. Inpaint + embed ────────────────────────────────────────────────────
    print('\n  [B] Inpainting batch ...')
    deid_outputs = run_inpaint_batch(
        batch, cfg.get('inpainting', {}), args.device
    )

    print('\n  [B] Embedding inpainted outputs ...')
    deid_embeddings = reid.embed_batch(deid_outputs)

    # ── Compute AUCs ─────────────────────────────────────────────────────────
    print('\n  Computing AUCs ...')

    # A: original vs original (self-similarity ceiling)
    auc_orig_orig = reid._compute_same_patient_auc(orig_embeddings, orig_embeddings)

    # B: inpainted vs original
    auc_deid_orig = reid._compute_same_patient_auc(orig_embeddings, deid_embeddings)

    gap = round(auc_orig_orig - auc_deid_orig, 4)

    print(f'\n  AUC original→original:  {auc_orig_orig:.4f}  (ceiling — should be ~1.0)')
    print(f'  AUC inpainted→original: {auc_deid_orig:.4f}  (actual pipeline privacy effect)')
    print(f'  Gap (de-id effect):     {gap:.4f}')
    print(f'  Target gap needed:      ≥ {round(auc_orig_orig - 0.55, 4)} (to reach AUC ≤ 0.55)')

    if gap < 0.05:
        verdict = (
            f'❌ CRITICAL: Gap of {gap:.4f} confirms the pipeline is providing almost no '
            f'de-identification. The inpainted images remain as recognisable as the originals '
            f'to RETFound. Architectural change required — expanded-region inpainting, full '
            f'synthesis, or adversarial perturbation pass.'
        )
    elif gap < 0.20:
        verdict = (
            f'⚠️  MARGINAL: Gap of {gap:.4f}. Pipeline is having some de-id effect but '
            f'nowhere near enough to reach AUC ≤ 0.55 target.'
        )
    else:
        verdict = (
            f'⚠️  PARTIAL: Gap of {gap:.4f}. Meaningful de-id effect but target not met.'
        )

    print(f'\n  {verdict}')

    summary = {
        'diagnostic': 'diag2_baseline',
        'date': date.today().isoformat(),
        'n_images': len(originals),
        'auc_original_vs_original': round(float(auc_orig_orig), 4),
        'auc_inpainted_vs_original': round(float(auc_deid_orig), 4),
        'gap': gap,
        'target_gap_needed': round(float(auc_orig_orig) - 0.55, 4),
        'verdict': verdict,
    }
    save_json(out_dir / 'diag2_baseline_results.json', summary)
    print(f'  Saved → {out_dir}')
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic 3 — FID Curve Across Conditioning Scales
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_SCALES = [0.4, 0.5, 0.6, 0.7]


def diag3_fid_curve(args, cfg, out_root, batch=None):
    """
    Run all four conditioning scales on the same N images.
    Report per-scale: FID, SSIM, LPIPS, same-patient AUC.

    Shows whether 0.5 was genuinely better or if the sweep was noise,
    and whether lower scales traded realism for privacy (the right trade-off)
    or just degraded both.
    """
    print('\n' + '='*60)
    print('DIAGNOSTIC 3 — FID Curve Across Conditioning Scales')
    print('='*60)

    out_dir = Path(out_root) / 'diag3_fid_curve'
    out_dir.mkdir(parents=True, exist_ok=True)

    if batch is None:
        image_paths = find_images(args.input, args.n_images)
        segmentation.load_model(args.weights)
        inpainting.load_model(cfg=cfg.get('inpainting', {}), device=args.device)
        batch = preprocess_batch(image_paths, cfg)

    originals = originals_dict(batch)
    orig_list  = list(originals.values())

    # Load RETFound if available (for per-scale AUC)
    retfound_available = False
    if args.retfound_weights:
        if args.retfound_dir and args.retfound_dir not in sys.path:
            sys.path.insert(0, args.retfound_dir)
        try:
            reid.load_retfound(
                weights_path=args.retfound_weights,
                retfound_dir=args.retfound_dir,
                device=args.device,
            )
            orig_embeddings = reid.embed_batch(originals)
            retfound_available = True
            print('  RETFound loaded — will compute per-scale AUC.')
        except Exception as e:
            print(f'  ⚠️  RETFound load failed: {e}. Skipping per-scale AUC.')
    else:
        print('  --retfound-weights not provided. FID + SSIM/LPIPS only (no per-scale AUC).')

    scale_results = []

    for scale in SWEEP_SCALES:
        print(f'\n  Scale {scale} ...')
        scale_dir = out_dir / f'scale_{scale}'
        scale_dir.mkdir(exist_ok=True)

        # Inpaint
        deid_outputs = run_inpaint_batch(
            batch, cfg.get('inpainting', {}), args.device,
            controlnet_conditioning_scale=scale,
        )

        if not deid_outputs:
            print(f'  ⚠️  No outputs for scale {scale}, skipping.')
            continue

        # Save outputs
        for stem, img in deid_outputs.items():
            cv2.imwrite(
                str(scale_dir / f'{stem}_scale{scale}.png'),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

        deid_list = list(deid_outputs.values())

        # FID
        print(f'    Computing FID ...')
        fid = realism.compute_fid(orig_list, deid_list, device=args.device)

        # SSIM + LPIPS
        print(f'    Computing SSIM/LPIPS ...')
        real_metrics = realism.compute_ssim_lpips(originals, deid_outputs)

        row = {
            'scale':      scale,
            'fid':        fid,
            'ssim_mean':  real_metrics.get('ssim_mean'),
            'lpips_mean': real_metrics.get('lpips_mean'),
            'n_images':   len(deid_outputs),
        }

        # Per-scale same-patient AUC
        if retfound_available:
            print(f'    Computing same-patient AUC ...')
            deid_embeddings = reid.embed_batch(deid_outputs)
            auc = reid._compute_same_patient_auc(orig_embeddings, deid_embeddings)
            row['same_patient_auc'] = round(float(auc), 4)
        else:
            row['same_patient_auc'] = None

        scale_results.append(row)

        auc_str = f"  AUC={row['same_patient_auc']:.4f}" if row['same_patient_auc'] else ''
        print(
            f'    scale={scale}: FID={fid}  SSIM={row["ssim_mean"]:.3f}'
            f'  LPIPS={row["lpips_mean"]:.3f}{auc_str}'
        )

    # ── Analysis ──────────────────────────────────────────────────────────────
    fids  = [r['fid'] for r in scale_results if r['fid'] is not None]
    fid_range = max(fids) - min(fids) if len(fids) > 1 else 0

    if fid_range <= 10:
        fid_verdict = (
            f'FID range across scales: {fid_range:.1f} (≤10). '
            f'Sweep was noise — conditioning scale does not meaningfully affect realism. '
            f'Look elsewhere for FID gains.'
        )
    else:
        best = min(scale_results, key=lambda r: r['fid'] or 999)
        fid_verdict = (
            f'FID range across scales: {fid_range:.1f}. '
            f'Best scale by FID: {best["scale"]} (FID={best["fid"]}).'
        )

    # Privacy vs realism trade-off analysis
    tradeoff_notes = []
    if retfound_available and len(scale_results) >= 2:
        # Lower scale = less ControlNet adherence = more generative freedom
        low  = next((r for r in scale_results if r['scale'] == 0.4), None)
        high = next((r for r in scale_results if r['scale'] == 0.7), None)
        if low and high and low['same_patient_auc'] and high['same_patient_auc']:
            auc_delta = high['same_patient_auc'] - low['same_patient_auc']
            fid_delta = (low['fid'] or 0) - (high['fid'] or 0)
            if low['same_patient_auc'] < high['same_patient_auc'] and fid_delta > 5:
                tradeoff_notes.append(
                    f'Scale 0.4 vs 0.7: AUC {high["same_patient_auc"]:.3f}→{low["same_patient_auc"]:.3f} '
                    f'(privacy improves by {auc_delta:.3f}) but FID worse by {fid_delta:.1f}. '
                    f'Lower scale trades realism for privacy — correct direction but not enough.'
                )
            elif auc_delta < 0.02:
                tradeoff_notes.append(
                    f'AUC barely changes across scales (delta={auc_delta:.3f}). '
                    f'Conditioning scale does not drive privacy. Confirms background texture '
                    f'is the biometric, not vessel pattern fidelity.'
                )

    print(f'\n  {fid_verdict}')

    summary = {
        'diagnostic': 'diag3_fid_curve',
        'date': date.today().isoformat(),
        'n_images': len(batch),
        'scales_tested': SWEEP_SCALES,
        'fid_range': round(fid_range, 2),
        'fid_verdict': fid_verdict,
        'tradeoff_notes': tradeoff_notes,
        'per_scale': scale_results,
    }
    save_json(out_dir / 'diag3_fid_curve_results.json', summary)
    print(f'  Saved → {out_dir}')
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(out_root, d1, d2, d3):
    lines = [
        '=' * 60,
        'REALIZEMED — DIAGNOSTIC SUMMARY',
        f'Date: {date.today().isoformat()}',
        '=' * 60,
        '',
    ]

    if d1:
        lines += [
            'DIAG 1 — LoRA Load Verification',
            f'  Mean MAD:       {d1["mean_mad"]}',
            f'  LoRA active:    {"YES ✅" if d1["lora_confirmed"] else "NO ❌"}',
            f'  Verdict:        {d1["verdict"]}',
            '',
        ]

    if d2:
        lines += [
            'DIAG 2 — No-Inpaint Privacy Baseline',
            f'  AUC orig→orig:      {d2["auc_original_vs_original"]}  (ceiling)',
            f'  AUC inpaint→orig:   {d2["auc_inpainted_vs_original"]}  (actual effect)',
            f'  Gap (de-id effect): {d2["gap"]}  '
            f'(need ≥{d2["target_gap_needed"]} to reach target)',
            f'  Verdict: {d2["verdict"]}',
            '',
        ]

    if d3:
        lines += [
            'DIAG 3 — FID Curve Across Conditioning Scales',
            f'  FID range:  {d3["fid_range"]}',
        ]
        for r in d3.get('per_scale', []):
            auc_str = f"  AUC={r['same_patient_auc']:.4f}" if r.get('same_patient_auc') else ''
            lines.append(
                f'  scale={r["scale"]}: FID={r["fid"]}  '
                f'SSIM={r["ssim_mean"]:.3f}  LPIPS={r["lpips_mean"]:.3f}{auc_str}'
            )
        lines += [f'  {d3["fid_verdict"]}']
        for note in d3.get('tradeoff_notes', []):
            lines.append(f'  {note}')
        lines.append('')

    # Next steps
    lines += [
        '=' * 60,
        'NEXT STEP — ARCHITECTURAL DECISION',
        '=' * 60,
        'Options on the table (Adam brief, 2026-06-08):',
        '  A. Expanded-region inpainting       ~1 week',
        '  B. Full synthesis                   ~1.5–2 weeks',
        '  C. Path C + adversarial privacy pass ~3–5 days  ← Adam lean',
        '',
        'Decision gate: review diagnostics_summary.txt + per-diag JSON',
        'then ping Adam directly.',
        '=' * 60,
    ]

    txt = '\n'.join(lines)
    summary_path = Path(out_root) / 'diagnostics_summary.txt'
    summary_path.write_text(txt)
    print('\n' + txt)
    print(f'\nSummary saved → {summary_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg['inpainting']['device'] = args.device

    run_lora      = args.all or args.diag == 'lora'
    run_baseline  = args.all or args.diag == 'baseline'
    run_fid_curve = args.all or args.diag == 'fid_curve'

    d1 = d2 = d3 = None

    # ── Diag 1 runs independently (different N, reloads models) ───────────────
    if run_lora:
        d1 = diag1_lora(args, cfg, out_root)

    # ── Diag 2 + 3 share a preprocessing batch ────────────────────────────────
    if run_baseline or run_fid_curve:
        print('\nPre-loading models for Diag 2 + 3 ...')
        segmentation.load_model(args.weights)
        inpainting.load_model(cfg=cfg.get('inpainting', {}), device=args.device)

        image_paths = find_images(args.input, args.n_images)
        print(f'Images for Diag 2 + 3: {len(image_paths)}')
        batch = preprocess_batch(image_paths, cfg)

        if run_baseline:
            d2 = diag2_baseline(args, cfg, out_root, batch=batch)

        if run_fid_curve:
            d3 = diag3_fid_curve(args, cfg, out_root, batch=batch)

    if args.all:
        write_summary(out_root, d1, d2, d3)


if __name__ == '__main__':
    main()
