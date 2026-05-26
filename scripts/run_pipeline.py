"""
scripts/run_pipeline.py
Batch de-identification on a folder of fundus images.

Usage:
    python scripts/run_pipeline.py \
        --input  /data/eyepacs/images/ \
        --output /data/deid_output/ \
        --weights /workspace/models/attention_unet/AttentionUNet.h5 \
        --config  configs/default.yaml \
        --device  cuda
"""

import argparse
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Make pipeline importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import preprocessing, segmentation, pathology, masking, inpainting


def parse_args():
    p = argparse.ArgumentParser(description='RealizeMD de-identification pipeline')
    p.add_argument('--input',   required=True, help='Folder of fundus images')
    p.add_argument('--output',  required=True, help='Output folder for de-identified images')
    p.add_argument('--weights', default=None,  help='Path to Model A .h5 weights')
    p.add_argument('--config',  default='configs/default.yaml')
    p.add_argument('--device',  default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--n', '--limit', type=int, default=None, dest='n', help='Limit to N images (for testing)')
    p.add_argument('--save-masks', action='store_true', help='Also save intermediate masks')
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg.setdefault('inpainting', {})['device'] = args.device

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
    if args.save_masks:
        mask_dir = output_dir / 'masks'
        mask_dir.mkdir(exist_ok=True)

    # Load models
    # Resolve weight path relative to project root
    weights_path = args.weights or cfg.get('segmentation', {}).get('weights')
    if not weights_path:
        raise ValueError('Provide --weights or set segmentation.weights in config')
    # Convert to absolute path if needed
    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        # Assume path is relative to the repository root
        weights_path = Path(__file__).resolve().parents[1] / weights_path
    print(f"🔧 Using segmentation weights at: {weights_path}")
    # Load segmentation model
    segmentation.load_model(str(weights_path))
    # Load inpainting model
    inpainting.load_model(
        cfg.get('inpainting', {}).get('weights', ''),
        device=args.device
    )

    # Reference image for histogram normalisation
    reference_rgb = preprocessing.select_reference(image_paths)

    # Process
    seg_cfg  = cfg.get('segmentation', {})
    pp_cfg   = cfg.get('preprocessing', {})
    mask_cfg = cfg.get('vessel_mask', {})
    path_cfg = cfg.get('pathology', {})
    inp_cfg  = cfg.get('inpainting', {})

    failed = []
    for img_path in tqdm(image_paths, desc='De-identifying'):
        try:
            # Preprocess
            preprocessed = preprocessing.preprocess(
                img_path,
                target_size=pp_cfg.get('target_size', 512),
                reference_rgb=reference_rgb,
            )

            # Segment
            vessel_mask = segmentation.predict(
                preprocessed,
                threshold=seg_cfg.get('threshold', 0.5)
            )

            # Detect pathology
            clahe_img = preprocessed['enhanced_rgb']
            lesion_result = pathology.detect_all(clahe_img, path_cfg)

            # Build mask
            mask_result = masking.build_inpaint_mask(
                vessel_mask=vessel_mask,
                lesion_mask=lesion_result['combined'],
                vessel_dilation_kernel=mask_cfg.get('dilation_kernel', 5),
            )
            st = mask_result['stats']
            tqdm.write(
                f"  [{img_path.name}] Vessel mask: {st['vessel_pct']}% | "
                f"Dilated: {st['vessel_dilated_pct']}% | Inpaint: {st['inpaint_pct']}%"
            )

            # Inpaint
            deid = inpainting.inpaint(
                image_rgb=preprocessed['original_rgb'],
                mask=mask_result['inpaint_mask'],
                device=inp_cfg.get('device', 'cuda'),
            )

            # Save
            out_path = output_dir / f'{img_path.stem}_deid.png'
            cv2.imwrite(str(out_path), cv2.cvtColor(deid, cv2.COLOR_RGB2BGR))

            if args.save_masks:
                cv2.imwrite(str(mask_dir / f'{img_path.stem}_vessel.png'), vessel_mask)
                cv2.imwrite(str(mask_dir / f'{img_path.stem}_inpaint.png'), mask_result['inpaint_mask'])
                cv2.imwrite(str(mask_dir / f'{img_path.stem}_lesion.png'), lesion_result['combined'])

        except Exception as e:
            print(f'\n  ⚠️  Failed {img_path.name}: {e}')
            failed.append(img_path.name)

    print(f'\n✅ Done. {len(image_paths) - len(failed)}/{len(image_paths)} images processed.')
    print(f'Output: {output_dir}')
    if failed:
        print(f'Failed ({len(failed)}): {failed[:5]}{"..." if len(failed) > 5 else ""}')


if __name__ == '__main__':
    main()
