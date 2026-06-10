"""
scripts/lora_ablation.py

Targeted LoRA ablation — rules out Adam's two failure modes before
building the adversarial perturbation pass on top of LoRA.

Failure mode (a): LoRA loaded but at near-zero effective scale due to
    adapter key prefix mismatch. Some diffusers versions silently succeed
    on load_lora_weights even when no keys match, resulting in fuse_lora
    being a no-op.

Failure mode (b): The no-LoRA control in Diag 1 was accidentally
    still-LoRA due to model state / caching. The _sd_pipe = None reset
    was in the script but worth confirming it fully reloaded.

What this script does:
    1. Load pipe WITH LoRA. Print all adapter key names + count.
       Check prefix alignment against expected UNet attention layer pattern.
    2. Run N images with fixed seed → save outputs.
    3. Reload pipe cleanly (no LoRA). Confirm _sd_pipe is a fresh object.
    4. Run same N images, same seeds → save outputs.
    5. Compare: per-image MAD, cosine sim in pixel space, SSIM.
    6. Optionally embed both sets with RETFound and report AUC + FID gap.
    7. Print verdict on both failure modes.

Usage:
    python scripts/lora_ablation.py \
        --input   /workspace/data/eyepacs/images/ \
        --output  /workspace/data/diagnostics/lora_ablation/ \
        --weights models/attention_unet/retina_attentionUnet_150epochs.hdf5 \
        --config  configs/default.yaml \
        --device  cuda \
        --n       5 \
        --seed    42

    # With RETFound (recommended — gives AUC gap as direction-of-effect signal)
    python scripts/lora_ablation.py ... \
        --retfound-weights models/RETFound_mae_natureCFP.pth \
        --retfound-dir     /workspace/RETFound

Outputs:
    lora_ablation/
        {stem}_lora_on.png
        {stem}_lora_off.png
        {stem}_diff_5x.png          # amplified pixel diff
        {stem}_compare.png          # original | on | off | diff side-by-side
        adapter_keys.txt            # full list of LoRA adapter key names
        lora_ablation_results.json
        lora_ablation_verdict.txt   # one-page summary for Adam
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

from pipeline import preprocessing, segmentation, masking, pathology
import pipeline.inpainting as inpainting_mod


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RealizeMD LoRA ablation')
    p.add_argument('--input',   required=True)
    p.add_argument('--output',  required=True)
    p.add_argument('--weights', required=True, help='Segmentation model .h5')
    p.add_argument('--config',  default='configs/default.yaml')
    p.add_argument('--device',  default='cuda')
    p.add_argument('--n',       type=int, default=5, help='Number of images (default 5)')
    p.add_argument('--seed',    type=int, default=42, help='Fixed seed for both runs')
    p.add_argument('--retfound-weights', default=None)
    p.add_argument('--retfound-dir',     default='/workspace/RETFound')
    return p.parse_args()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, cls=NumpyEncoder))


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode (a) — key prefix check
# ─────────────────────────────────────────────────────────────────────────────

# Expected pattern: LoRA keys for SD UNet attention layers should contain these
EXPECTED_KEY_FRAGMENTS = [
    'unet',
    'attn',
]

def check_adapter_keys(pipe, out_dir):
    """
    Inspect LoRA adapter keys loaded into the pipe.
    Returns (keys, n_matched, alignment_ok).
    """
    keys = []

    # diffusers >= 0.21: pipe.unet.attn_processors contains the LoRA layers
    try:
        attn_procs = pipe.unet.attn_processors
        # keys here are processor names, e.g. "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"
        keys = list(attn_procs.keys())
        source = 'unet.attn_processors'
    except AttributeError:
        pass

    # Fallback: named parameters containing 'lora'
    if not keys:
        keys = [
            name for name, _ in pipe.unet.named_parameters()
            if 'lora' in name.lower()
        ]
        source = 'unet.named_parameters (lora filter)'

    # Write full key list to file
    key_path = out_dir / 'adapter_keys.txt'
    key_path.write_text(
        f'Source: {source}\nTotal keys: {len(keys)}\n\n' +
        '\n'.join(keys)
    )

    # Check whether any keys contain expected fragments
    matched = [
        k for k in keys
        if all(frag in k for frag in EXPECTED_KEY_FRAGMENTS)
    ]

    alignment_ok = len(matched) > 0
    return keys, len(matched), alignment_ok, source


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode (b) — clean reload verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_clean_reload(cfg_no_lora, device):
    """
    Reload the pipeline with LoRA disabled.
    Returns the new _sd_pipe object id to confirm it's a fresh instance.
    """
    # Force clear
    inpainting_mod._sd_pipe = None
    assert inpainting_mod._sd_pipe is None, 'Failed to clear _sd_pipe'

    inpainting_mod.load_model(cfg=cfg_no_lora, device=device)

    assert inpainting_mod._sd_pipe is not None, 'load_model did not set _sd_pipe'
    pipe_id = id(inpainting_mod._sd_pipe['pipe'])
    return pipe_id


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_batch(image_paths, cfg):
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
                threshold=seg_cfg.get('threshold', 0.5),
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
                'path':         img_path,
                'stem':         img_path.stem,
                'preprocessed': preprocessed,
                'vessel_mask':  vessel_mask,
                'mask_result':  mask_result,
            })
        except Exception as e:
            print(f'  ⚠️  Pre-processing failed {img_path.name}: {e}')
    return batch


def run_inpaint_fixed_seed(batch, cfg, device, seed):
    """Inpaint all items in batch with the same fixed seed."""
    outputs = {}
    for item in tqdm(batch, desc='  Inpainting'):
        try:
            deid = inpainting_mod.inpaint(
                image_rgb=item['preprocessed']['original_rgb'],
                mask=item['mask_result']['inpaint_mask'],
                vessel_mask=item['vessel_mask'],
                device=device,
                seed=seed,
                fov=item['preprocessed']['fov'],
            )
            outputs[item['stem']] = deid
        except Exception as e:
            print(f'  ⚠️  Inpainting failed {item["path"].name}: {e}')
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def compare_outputs(outputs_on, outputs_off):
    """
    Per-image MAD, cosine similarity, SSIM between LoRA-on and LoRA-off outputs.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    results = []
    common = [s for s in outputs_on if s in outputs_off]

    for stem in common:
        on  = outputs_on[stem].astype(np.float32)
        off = outputs_off[stem].astype(np.float32)

        mad = float(np.mean(np.abs(on - off)))

        flat_on  = on.flatten()
        flat_off = off.flatten()
        cos_sim  = float(
            np.dot(flat_on, flat_off) /
            (np.linalg.norm(flat_on) * np.linalg.norm(flat_off) + 1e-8)
        )

        ssim_val = float(ssim_fn(
            outputs_on[stem], outputs_off[stem],
            channel_axis=2, data_range=255,
        ))

        results.append({
            'stem':    stem,
            'mad':     round(mad, 4),
            'cos_sim': round(cos_sim, 6),
            'ssim':    round(ssim_val, 4),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Optional RETFound comparison
# ─────────────────────────────────────────────────────────────────────────────

def retfound_comparison(outputs_on, outputs_off, retfound_weights,
                        retfound_dir, device):
    """
    Embed both sets and compute:
      - Mean cosine similarity between paired on/off embeddings
        (high = LoRA not moving the embedding; low = LoRA is effective)
      - Same-patient AUC: on vs originals, off vs originals
        to confirm direction-of-effect
    """
    if retfound_dir and retfound_dir not in sys.path:
        sys.path.insert(0, retfound_dir)

    from eval import reid

    reid.load_retfound(
        weights_path=retfound_weights,
        retfound_dir=retfound_dir,
        device=device,
    )

    emb_on  = reid.embed_batch(outputs_on)
    emb_off = reid.embed_batch(outputs_off)

    common = [s for s in emb_on if s in emb_off]
    cos_sims = []
    for stem in common:
        v_on  = emb_on[stem]
        v_off = emb_off[stem]
        cos   = float(
            np.dot(v_on, v_off) /
            (np.linalg.norm(v_on) * np.linalg.norm(v_off) + 1e-8)
        )
        cos_sims.append(cos)

    mean_emb_cos = float(np.mean(cos_sims)) if cos_sims else None

    return {
        'mean_embedding_cosine_on_vs_off': round(mean_emb_cos, 4) if mean_emb_cos else None,
        'interpretation': (
            'Embeddings nearly identical — LoRA not moving RETFound representation'
            if mean_emb_cos and mean_emb_cos > 0.98 else
            'Embeddings differ — LoRA is shifting the representation ✅'
            if mean_emb_cos and mean_emb_cos < 0.95 else
            f'Marginal embedding shift (cos={mean_emb_cos:.4f}) — LoRA effect weak'
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save comparison images
# ─────────────────────────────────────────────────────────────────────────────

def save_comparisons(batch, outputs_on, outputs_off, out_dir):
    for item in batch:
        stem = item['stem']
        if stem not in outputs_on or stem not in outputs_off:
            continue

        orig    = item['preprocessed']['original_rgb']
        img_on  = outputs_on[stem]
        img_off = outputs_off[stem]

        # Amplified diff (×5)
        diff_amp = np.clip(
            np.abs(img_on.astype(np.float32) - img_off.astype(np.float32)) * 5,
            0, 255
        ).astype(np.uint8)

        # Individual saves
        cv2.imwrite(str(out_dir / f'{stem}_lora_on.png'),
                    cv2.cvtColor(img_on, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f'{stem}_lora_off.png'),
                    cv2.cvtColor(img_off, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f'{stem}_diff_5x.png'),
                    cv2.cvtColor(diff_amp, cv2.COLOR_RGB2BGR))

        # Side-by-side: original | lora_on | lora_off | diff×5
        row = np.concatenate([orig, img_on, img_off, diff_amp], axis=1)
        cv2.imwrite(str(out_dir / f'{stem}_compare.png'),
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

def build_verdict(key_check, clean_reload_confirmed, per_image,
                  retfound_result=None):
    mean_mad  = float(np.mean([r['mad']  for r in per_image]))
    mean_ssim = float(np.mean([r['ssim'] for r in per_image]))

    # ── Failure mode (a): key prefix alignment ────────────────────────────────
    keys, n_matched, alignment_ok, source = key_check
    if not alignment_ok:
        fm_a = (
            f'❌ FAILURE MODE (a) CONFIRMED: LoRA adapter keys ({len(keys)} total) '
            f'contain no UNet attention layer matches. fuse_lora was a no-op. '
            f'Key source: {source}. Check lora_weights path and that the LoRA '
            f'was trained against the same SD base model.'
        )
    else:
        fm_a = (
            f'✅ Failure mode (a) clear: {n_matched}/{len(keys)} adapter keys '
            f'matched expected UNet attention pattern. Key source: {source}.'
        )

    # ── Failure mode (b): clean reload ────────────────────────────────────────
    if clean_reload_confirmed:
        fm_b = '✅ Failure mode (b) clear: _sd_pipe was None before no-LoRA reload. Fresh instance confirmed.'
    else:
        fm_b = '❌ FAILURE MODE (b) CONFIRMED: _sd_pipe was not properly cleared. No-LoRA control was contaminated.'

    # ── Direction-of-effect ───────────────────────────────────────────────────
    if mean_mad < 1.5:
        effect = (
            f'⚠️  Low pixel MAD ({mean_mad:.2f}). LoRA influence may be primarily '
            f'textural/frequency-domain rather than structural — low MAD does not '
            f'necessarily mean LoRA is inactive. Check diff images visually and '
            f'RETFound embedding cosine similarity if available.'
        )
    elif mean_mad < 5.0:
        effect = (
            f'⚠️  Moderate pixel MAD ({mean_mad:.2f}). LoRA is having some effect '
            f'but influence is subtle. Consider raising lora_scale above 0.7 or '
            f'retraining at higher step count.'
        )
    else:
        effect = (
            f'✅ Strong pixel MAD ({mean_mad:.2f}). LoRA is meaningfully changing outputs.'
        )

    # ── RETFound signal ───────────────────────────────────────────────────────
    rf_line = ''
    if retfound_result:
        rf_line = f'\nRETFound: {retfound_result["interpretation"]}'

    verdict = '\n'.join([
        '=' * 60,
        'LORA ABLATION VERDICT',
        f'Date: {date.today().isoformat()}',
        '=' * 60,
        '',
        fm_a,
        '',
        fm_b,
        '',
        f'Direction of effect (pixel space):',
        f'  Mean MAD:  {mean_mad:.4f}',
        f'  Mean SSIM: {mean_ssim:.4f}  (1.0 = identical)',
        f'  {effect}',
        rf_line,
        '',
        '=' * 60,
    ])

    return verdict, {
        'failure_mode_a_clear': alignment_ok,
        'failure_mode_b_clear': clean_reload_confirmed,
        'n_adapter_keys':       len(keys),
        'n_matched_keys':       n_matched,
        'mean_mad':             round(mean_mad, 4),
        'mean_ssim':            round(mean_ssim, 4),
        'retfound':             retfound_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['inpainting']['device'] = args.device

    # Find images
    image_paths = sorted(
        list(Path(args.input).rglob('*.jpeg')) +
        list(Path(args.input).rglob('*.jpg')) +
        list(Path(args.input).rglob('*.png'))
    )[:args.n]
    if not image_paths:
        raise FileNotFoundError(f'No images found in {args.input}')
    print(f'Images: {len(image_paths)}  |  Seed: {args.seed}')

    # Pre-process once — shared by both runs
    segmentation.load_model(args.weights)
    batch = preprocess_batch(image_paths, cfg)

    # ── RUN 1: WITH LoRA ──────────────────────────────────────────────────────
    print('\n[1/2] Loading pipeline WITH LoRA ...')
    inpainting_mod._sd_pipe = None
    inpainting_mod.load_model(cfg=cfg['inpainting'], device=args.device)

    # Key prefix check (failure mode a)
    pipe_obj = inpainting_mod._sd_pipe['pipe']
    key_check = check_adapter_keys(pipe_obj, out_dir)
    _, n_matched, alignment_ok, source = key_check
    print(f'  Adapter keys: {len(key_check[0])} total, {n_matched} matched UNet attn pattern')
    print(f'  Alignment OK: {"✅" if alignment_ok else "❌ — fuse_lora may be no-op"}')

    pipe_id_lora_on = id(inpainting_mod._sd_pipe['pipe'])
    outputs_on = run_inpaint_fixed_seed(batch, cfg['inpainting'], args.device, args.seed)

    # ── RUN 2: WITHOUT LoRA ───────────────────────────────────────────────────
    print('\n[2/2] Reloading pipeline WITHOUT LoRA ...')
    cfg_no_lora = copy.deepcopy(cfg)
    cfg_no_lora['inpainting']['lora_weights'] = None

    pipe_id_no_lora = verify_clean_reload(cfg_no_lora['inpainting'], args.device)
    clean_reload_confirmed = (pipe_id_no_lora != pipe_id_lora_on)
    print(f'  Clean reload: {"✅ fresh instance" if clean_reload_confirmed else "❌ same object — state leak"}')

    outputs_off = run_inpaint_fixed_seed(batch, cfg_no_lora['inpainting'], args.device, args.seed)

    # ── Compare ───────────────────────────────────────────────────────────────
    print('\nComparing outputs ...')
    per_image = compare_outputs(outputs_on, outputs_off)
    for r in per_image:
        print(f'  {r["stem"]}: MAD={r["mad"]:.2f}  cos_sim={r["cos_sim"]:.6f}  SSIM={r["ssim"]:.4f}')

    # ── RETFound (optional) ───────────────────────────────────────────────────
    retfound_result = None
    if args.retfound_weights:
        print('\nRETFound embedding comparison ...')
        try:
            retfound_result = retfound_comparison(
                outputs_on, outputs_off,
                args.retfound_weights, args.retfound_dir, args.device,
            )
            print(f'  {retfound_result["interpretation"]}')
        except Exception as e:
            print(f'  ⚠️  RETFound comparison failed: {e}')

    # ── Save comparison images ────────────────────────────────────────────────
    print('\nSaving comparison images ...')
    save_comparisons(batch, outputs_on, outputs_off, out_dir)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict_txt, verdict_data = build_verdict(
        key_check, clean_reload_confirmed, per_image, retfound_result
    )

    print('\n' + verdict_txt)

    verdict_data['per_image'] = per_image
    verdict_data['date'] = date.today().isoformat()
    verdict_data['seed'] = args.seed
    verdict_data['n_images'] = len(per_image)

    save_json(out_dir / 'lora_ablation_results.json', verdict_data)
    (out_dir / 'lora_ablation_verdict.txt').write_text(verdict_txt)
    print(f'Saved → {out_dir}')


if __name__ == '__main__':
    main()
