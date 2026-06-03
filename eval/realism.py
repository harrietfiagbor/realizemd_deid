"""
eval/realism.py
Realism metrics: FID, SSIM, LPIPS.
Implements Section 2.3 of the supervisor's eval framework.
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def compute_ssim_lpips(original_images: dict,
                       deid_images: dict) -> dict:
    """
    Per-image SSIM and LPIPS between original and de-identified images.

    Target SSIM window: 0.65–0.85
      Too high = under-de-identified (not enough changed)
      Too low  = unrealistic (too much changed)

    Target LPIPS: 0.15–0.35
    """
    from skimage.metrics import structural_similarity as ssim_fn
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')
        use_lpips = True
    except ImportError:
        print('  ⚠️  lpips not installed, skipping LPIPS')
        use_lpips = False

    results = []
    stems = [s for s in deid_images if s in original_images]

    for stem in tqdm(stems, desc='SSIM/LPIPS'):
        orig = original_images[stem]
        deid = deid_images[stem]

        if orig.shape != deid.shape:
            deid = cv2.resize(deid, (orig.shape[1], orig.shape[0]))

        # SSIM — computed on RGB (channel_axis=2) so colour shifts
        # (e.g. global blue cast from inpainting) are correctly penalised.
        # Y-channel SSIM is intentionally avoided: it misses colour artifacts
        # and produced a false-positive 0.85 on the LaMa blue-blob failure.
        ssim_val = float(ssim_fn(orig, deid, data_range=255, channel_axis=2))

        row = {'stem': stem, 'ssim': round(ssim_val, 4)}

        # LPIPS
        if use_lpips:
            import torch
            orig_t = torch.from_numpy(orig).permute(2,0,1).float() / 127.5 - 1
            deid_t = torch.from_numpy(deid).permute(2,0,1).float() / 127.5 - 1
            lpips_val = float(lpips_fn(orig_t.unsqueeze(0), deid_t.unsqueeze(0)))
            row['lpips'] = round(lpips_val, 4)

        results.append(row)

    import pandas as pd
    df = pd.DataFrame(results)

    ssim_pass = (
        (df.ssim >= 0.65) & (df.ssim <= 0.85)
    ).mean()

    summary = {
        'ssim_mean':     round(df.ssim.mean(), 4),
        'ssim_std':      round(df.ssim.std(), 4),
        'ssim_in_window': round(ssim_pass, 4),
        'ssim_pass':     (0.65 <= df.ssim.mean() <= 0.85),
        'per_image':     results,
    }

    if 'lpips' in df.columns:
        summary.update({
            'lpips_mean': round(df.lpips.mean(), 4),
            'lpips_std':  round(df.lpips.std(), 4),
            'lpips_pass': (0.15 <= df.lpips.mean() <= 0.35),
        })

    return summary


def compute_fid(real_images: list,
                deid_images: list,
                device: str = 'cuda') -> float:
    """
    Fréchet Inception Distance between real and de-identified image sets.
    Target: < 30 (good), < 15 (excellent).

    Args:
        real_images: list of np.ndarray (H, W, 3) uint8
        deid_images: list of np.ndarray (H, W, 3) uint8
    """
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        import tempfile, os

        # Save to temp dirs — pytorch_fid expects folder paths
        with tempfile.TemporaryDirectory() as real_dir, \
             tempfile.TemporaryDirectory() as deid_dir:

            for i, img in enumerate(real_images):
                cv2.imwrite(
                    os.path.join(real_dir, f'{i:05d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
            for i, img in enumerate(deid_images):
                cv2.imwrite(
                    os.path.join(deid_dir, f'{i:05d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )

            fid = calculate_fid_given_paths(
                [real_dir, deid_dir],
                batch_size=32,
                device=device,
                dims=2048,
            )
        return round(float(fid), 2)

    except ImportError:
        print('  ⚠️  pytorch_fid not installed. Run: pip install pytorch-fid')
        return None
