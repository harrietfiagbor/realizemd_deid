"""
vae_only_spot_baseline.py

Task 2 of the plan: Adam's ask for the "nothing invented" reference point --
run the healthy-eye false-spot detector on VAE-only round-trips (no UNet at all)
of the 8 healthy images. This is the baseline the 62-false-spots-at-vessel-mask-
scale-1.75 number should be judged against.

Reuses vae_roundtrip() and the load/vessel functions from vessel_metrics_reference.py
(the reference script) and count_false_spots() ported directly from
conditioned_reconstruction_eval.ipynb, same approximation caveat as before: this is
NOT Fiifi's exact spot-detection code, it's a reasonable reconstruction of it
(per-pixel new-bright-material blob count), so treat these numbers as directional,
not a precise match to her Test 6 baseline.

RunPod usage:
    python scripts/vae_only_spot_baseline.py \
        /workspace/diffusion_retina/conditioned_reconstructions \
        --out /workspace/diffusion_retina_victoria/vae_only_baseline
"""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "."
sys.path.insert(0, REPO_ROOT)
from vessel_metrics_reference import load_rgb, vae_roundtrip, HEALTHY

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi


def count_false_spots(orig_rgb, recon_rgb, min_area=5, max_area=800, diff_thresh=25):
    """Same implementation as conditioned_reconstruction_eval.ipynb -- ported
    verbatim, not reimplemented from memory, to avoid a silent drift bug."""
    diff = np.abs(orig_rgb.astype(int) - recon_rgb.astype(int)).sum(axis=2)
    new_bright = (recon_rgb.astype(int).sum(axis=2) - orig_rgb.astype(int).sum(axis=2) > diff_thresh) & (diff > diff_thresh)
    labeled, n = ndi.label(new_bright)
    count = 0
    for i in range(1, n + 1):
        area = (labeled == i).sum()
        if min_area <= area <= max_area:
            count += 1
    return count, new_bright


def main():
    import argparse
    from diffusers import AutoencoderKL

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="Folder with {name}_original.png (healthy set)")
    parser.add_argument("--out", required=True, help="Output directory for VAE-only images + overlays")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("sd2-community/stable-diffusion-2-1-base", subfolder="vae")
    vae.eval()

    rows = []
    for name in HEALTHY:
        orig = load_rgb(src_dir / f"{name}_original.png")
        vae_out = vae_roundtrip(orig, vae)
        n_spots, spot_mask = count_false_spots(orig, vae_out)

        cv2.imwrite(str(out_dir / f"{name}_vae_only.png"), cv2.cvtColor(vae_out, cv2.COLOR_RGB2BGR))
        overlay = vae_out.copy()
        overlay[spot_mask] = [255, 0, 0]
        cv2.imwrite(str(out_dir / f"{name}_spot_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        rows.append((name, n_spots))
        print(f"{name}: {n_spots} false spots (VAE-only round-trip)")

    counts = [c for _, c in rows]
    print(f"\nMean: {np.mean(counts):.1f}  Range: {min(counts)}-{max(counts)}")
    print(f"For comparison, vessel-mask-control@1.75 reported: 62 false spots")
    print(f"Images + overlays saved to {out_dir}/ for visual spot-check (plan Task 2 verify step)")


if __name__ == "__main__":
    main()
