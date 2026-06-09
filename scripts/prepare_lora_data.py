"""
scripts/prepare_lora_data.py
Prepares fundus images for LoRA fine-tuning.
Runs preprocessing (resize to 512x512, FOV mask) on N EyePACS images
and saves to output directory.

Usage:
    python scripts/prepare_lora_data.py \
        --input  /workspace/data/eyepacs/images/ \
        --output /workspace/fundus_train/ \
        --n 200
"""

import argparse
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import preprocessing


def parse_args():
    p = argparse.ArgumentParser(description='Prepare LoRA training data')
    p.add_argument('--input',  required=True, help='Folder of raw EyePACS images')
    p.add_argument('--output', required=True, help='Output folder for processed images')
    p.add_argument('--n',      type=int, default=200, help='Number of images to prepare')
    return p.parse_args()


def main():
    args = parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(src.rglob('*.jpg')) +
        list(src.rglob('*.jpeg')) +
        list(src.rglob('*.png'))
    )[:args.n]

    print(f'Processing {len(images)} images → {dst}')

    failed = []
    for p in tqdm(images, desc='Preparing'):
        try:
            result = preprocessing.preprocess(p, target_size=512)
            out = cv2.cvtColor(result['original_rgb'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dst / f'{p.stem}.png'), out)
        except Exception as e:
            tqdm.write(f'  ⚠️  {p.name}: {e}')
            failed.append(p.name)

    total = len(list(dst.glob('*.png')))
    print(f'\nDone. {total} images saved to {dst}')
    if failed:
        print(f'Failed ({len(failed)}): {failed[:5]}{"..." if len(failed) > 5 else ""}')


if __name__ == '__main__':
    main()
