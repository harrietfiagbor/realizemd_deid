"""
scripts/run_pipeline.py
Batch de-identification on a folder of fundus images.

Usage (standard):
    python scripts/run_pipeline.py \
        --input  /data/eyepacs/images/ \
        --output /data/deid_output/ \
        --weights /workspace/models/attention_unet/AttentionUNet.h5 \
        --config  configs/default.yaml \
        --device  cuda

Usage (conditioning scale sweep — run before full pilot):
    python scripts/run_pipeline.py \
        --input  /data/eyepacs/images/ \
        --output /data/sweep_output/ \
        --weights /workspace/models/attention_unet/AttentionUNet.h5 \
        --n 10 \
        --sweep
    Produces one subfolder per scale (scale_0.4/, scale_0.5/, etc.)
    Visually inspect + run eval to pick best scale before full 50-image pilot.
"""

import argparse
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import preprocessing, segmentation, pathology, masking, inpainting


SWEEP_SCALES = [0.4, 0.5, 0.6, 0.7]


def parse_args():
    p = argparse.ArgumentParser(description='RealizeMD de-identification pipeline')
    p.add_argument('--input',      required=True, help='Folder of fundus images')
    p.add_argument('--output',     required=True, help='Output folder for de-identified images')
    p.add_argument('--weights',    default=None,  help='Path to Model A .h5 weights')
    p.add_argument('--config',     default='configs/default.yaml')
    p.add_argument('--device',     default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--n', '--limit', type=int, default=None, dest='n',
                   help='Limit to N images')
    p.add_argument('--save-masks', action='store_true',
                   help='Also save intermediate masks')
    p.add_argument('--sweep',      action='store_true',
                   help=(
                       f'Sweep controlnet_conditioning_scale over {SWEEP_SCALES} '
                       f'on --n images (default 10). Saves one subfolder per scale. '
                       f'Use before full 50-image pilot to pick best scale.'
                   ))
    return p.parse_args()


def process_image(img_path, preprocessed, vessel_mask, lesion_result, mask_result,
                  inp_cfg, output_dir, mask_dir, dil, device,
                  controlnet_conditioning_scale=None, save_masks=False):
    """Run inpainting + save for one image. Returns output path or None on failure."""
    try:
        import torch
        seed = int(torch.randint(0, 2**31, (1,)).item())

        deid = inpainting.inpaint(
            image_rgb=preprocessed['original_rgb'],
            mask=mask_result['inpaint_mask'],
            vessel_mask=vessel_mask,
            device=inp_cfg.get('device', device),
            seed=seed,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        out_path = output_dir / f'{img_path.stem}_deid_dil{dil}.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(deid, cv2.COLOR_RGB2BGR))

        if save_masks and mask_dir is not None:
            cv2.imwrite(str(mask_dir / f'{img_path.stem}_vessel_dil{dil}.png'), vessel_mask)
            cv2.imwrite(str(mask_dir / f'{img_path.stem}_inpaint_dil{dil}.png'),
                        mask_result['inpaint_mask'])
            cv2.imwrite(str(mask_dir / f'{img_path.stem}_lesion_dil{dil}.png'),
                        lesion_result['combined'])

        return out_path
    except Exception as e:
        print(f'\n  ⚠️  Inpainting failed {img_path.name}: {e}')
        return None


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg.setdefault('inpainting', {})['device'] = args.device

    # If sweep mode and no --n given, default to 10 images
    if args.sweep and args.n is None:
        args.n = 10
        print(f'--sweep mode: defaulting to 10 images. Use --n to override.')

    # Find images
    input_dir = Path(args.input)
    image_paths = sorted(
        list(input_dir.rglob('*.jpeg')) +
        list(input_dir.rglob('*.jpg')) +
        list(input_dir.rglob('*.png'))
    )
    if args.n:
        image_paths = image_paths[:args.n]
    print(f'Found {len(image_paths)} images')

    # Output dirs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    weights_path = args.weights or cfg.get('segmentation', {}).get('weights')
    if not weights_path:
        raise ValueError('Provide --weights or set segmentation.weights in config')
    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        weights_path = Path(__file__).resolve().parents[1] / weights_path
    print(f'Segmentation weights: {weights_path}')
    segmentation.load_model(str(weights_path))

    inpainting.load_model(
        cfg=cfg.get('inpainting', {}),
        device=args.device,
    )

    reference_rgb = preprocessing.select_reference(image_paths)

    seg_cfg  = cfg.get('segmentation', {})
    pp_cfg   = cfg.get('preprocessing', {})
    mask_cfg = cfg.get('vessel_mask', {})
    path_cfg = cfg.get('pathology', {})
    inp_cfg  = cfg.get('inpainting', {})
    dil      = mask_cfg.get('dilation_kernel', 3)

    # ── Pre-process all images once (shared between sweep scales) ─────────────
    print('\nPre-processing images...')
    processed = []
    for img_path in tqdm(image_paths, desc='Pre-processing'):
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
            clahe_img = preprocessed['enhanced_rgb']
            lesion_result = pathology.detect_all(clahe_img, path_cfg)
            mask_result = masking.build_inpaint_mask(
                vessel_mask=vessel_mask,
                lesion_mask=lesion_result['combined'],
                vessel_dilation_kernel=dil,
            )
            st = mask_result['stats']
            tqdm.write(
                f'  [{img_path.name}] vessel={st["vessel_pct"]}% '
                f'dilated={st["vessel_dilated_pct"]}% inpaint={st["inpaint_pct"]}%'
            )
            processed.append((img_path, preprocessed, vessel_mask, lesion_result, mask_result))
        except Exception as e:
            print(f'\n  ⚠️  Pre-processing failed {img_path.name}: {e}')

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if args.sweep:
        print(f'\nSweep mode: testing conditioning scales {SWEEP_SCALES}')
        for scale in SWEEP_SCALES:
            scale_dir = output_dir / f'scale_{scale}'
            scale_dir.mkdir(exist_ok=True)
            mask_dir = (scale_dir / 'masks') if args.save_masks else None
            if mask_dir:
                mask_dir.mkdir(exist_ok=True)

            failed = []
            for img_path, preprocessed, vessel_mask, lesion_result, mask_result in tqdm(
                processed, desc=f'  scale={scale}'
            ):
                result = process_image(
                    img_path, preprocessed, vessel_mask, lesion_result, mask_result,
                    inp_cfg, scale_dir, mask_dir, dil, args.device,
                    controlnet_conditioning_scale=scale,
                    save_masks=args.save_masks,
                )
                if result is None:
                    failed.append(img_path.name)

            print(
                f'  scale={scale}: {len(processed) - len(failed)}/{len(processed)} done'
                + (f' | failed: {failed}' if failed else '')
            )

        print(f'\nSweep complete. Results in: {output_dir}')
        print('Next: visually inspect + run eval per subfolder, then set')
        print('  inpainting.controlnet_conditioning_scale in default.yaml')
        print('  and run the full 50-image pilot.')

    # ── Standard run ──────────────────────────────────────────────────────────
    else:
        mask_dir = (output_dir / 'masks') if args.save_masks else None
        if mask_dir:
            mask_dir.mkdir(exist_ok=True)

        failed = []
        for img_path, preprocessed, vessel_mask, lesion_result, mask_result in tqdm(
            processed, desc='Inpainting'
        ):
            result = process_image(
                img_path, preprocessed, vessel_mask, lesion_result, mask_result,
                inp_cfg, output_dir, mask_dir, dil, args.device,
                save_masks=args.save_masks,
            )
            if result is None:
                failed.append(img_path.name)

        print(f'\nDone. {len(processed) - len(failed)}/{len(processed)} images processed.')
        print(f'Output: {output_dir}')
        if failed:
            print(f'Failed ({len(failed)}): {failed[:5]}{"..." if len(failed) > 5 else ""}')


if __name__ == '__main__':
    main()
