"""
vessel_metrics_reference.py

REFERENCE IMPLEMENTATION OF RECORD for vessel-fidelity metrics on the diffusion
reconstruction pipeline (per Adam, 2026-10-05: "Victoria's script is the one of
record"). Harriet: please run this unmodified against your files rather than your
own implementation, so the two of us stop producing numbers that can't be
reconciled under the same definition.

This script, and this docstring, are the direct answer to the four things Adam
asked to have shared:

1. THE clDice SCRIPT: this file. cl_dice() below -- standard centerline Dice,
   skeletonize both masks, check each skeleton's containment in the OTHER mask
   (not skeleton-vs-skeleton), symmetric precision/sensitivity combined
   harmonically. No resizing, cropping, or smoothing applied to either mask
   before skeletonization.

2. PREPROCESSING STEPS: vessel_mask() below. Green-channel CLAHE only
   (clipLimit=2.0, tileGridSize=(8,8)) via Model A's own expected input
   contract, threshold=0.5 on the sigmoid output. No resize anywhere in this
   pipeline -- images are used at whatever resolution they're saved at (512x512
   for every file in conditioned_reconstructions/), and Model A's predict() is
   called directly on that, no intermediate resampling step exists to introduce
   a resolution mismatch.

3. THE 1PX/NOISE-FLOOR PERTURBATION: noise_floor_perturb() below --
   cv2.GaussianBlur(img, (3,3), sigmaX=0.6) applied to the raw original. Nothing
   else: no resize, no recompression, no additional noise. This is a blur-based
   floor, not a literal 1-pixel translation -- if Harriet's "1px shift" floor
   used a different perturbation (e.g. np.roll by 1 pixel), that is a candidate
   source of the two floor numbers (0.95 vs ~0.60) not matching, flag this
   explicitly when comparing results, don't assume it's the same operation.

4. DR vs. HEALTHY SPLIT: HEALTHY and DR lists below, exact names.
   DR set (5 images): IDRiD_03, IDRiD_09, IDRiD_10, IDRiD_11, IDRiD_25
   Healthy set (8 images): 10977_right, 10983_left, 10983_right, 10987_left,
   10987_right, 10988_left, 10988_right, 10989_left

Scope, explicit and non-negotiable for reconciliation purposes:
  - RAW FILES ONLY. Not matplotlib figure crops. If you're pointing this at a
    rendered comparison figure instead of the original/reconstructed PNGs, stop,
    that's exactly the discrepancy class we're trying to eliminate.
  - clDice is the metric of record (not pixel IoU, which punishes thin
    structures too harshly, and not the generic "vessel dice" SSIM style
    measure). Reported as (clDice, precision, sensitivity) per image, same
    weighting for every centerline point regardless of vessel caliber -- see
    the caliber-stratified analysis (thick ~88.5%, mid ~75.7%, thin ~57.9%
    recovery) for why whole-image clDice alone understates fidelity on the
    major vessel tree.

Source of truth this was extracted from: conditioned_reconstruction_eval.ipynb
(originally a notebook, not a standalone script -- ported here and re-verified
line-for-line to reproduce that notebook's numbers exactly before being handed
off, see scratch/vessel_metrics_cli_verify2.txt for the verification run).

Model A weights provenance: see scripts/verify_model_a_weights.py -- run it on
both sides and compare the printed MD5 before trusting "same model" for any
cross-machine comparison. Local reference MD5: 9cbd7640f82f083e9479240dcbb087cc
(not independently verified against the arkanivasarkar GitHub repo itself, only
confirms whether two parties are using the identical file as each other).

RunPod usage (this repo's checkout lives at /workspace/realizemd_deid, Model A
weights land at /workspace/models/attention_unet/ per scripts/setup_runpod.sh --
exact filename inside that folder not independently confirmed here, pass it
explicitly rather than trust a guessed default):

    python scripts/vessel_metrics_reference.py \
        /workspace/diffusion_retina/conditioned_reconstructions \
        --weights "/workspace/models/attention_unet/<actual filename>.h5" \
        --out /workspace/diffusion_retina_victoria/vessel_metrics_results.csv
"""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "."
sys.path.insert(0, REPO_ROOT)
from pipeline import segmentation

import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize

HEALTHY = ["10977_right", "10983_left", "10983_right", "10987_left", "10987_right",
           "10988_left", "10988_right", "10989_left"]
DR = ["IDRiD_03", "IDRiD_09", "IDRiD_10", "IDRiD_11", "IDRiD_25"]
ALL_NAMES = HEALTHY + DR


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def vessel_mask(img_rgb, threshold=0.5):
    """Model A vessel mask. img_rgb must already be the real image at its native
    size -- no resize here, Model A's own predict() handles that internally via
    the pipeline.segmentation module's expected input contract."""
    green_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_rgb[:, :, 1])
    pre = {"green_clahe": green_clahe, "enhanced_rgb": img_rgb}
    mask = segmentation.predict(pre, threshold=threshold)
    return mask > 0


def cl_dice(a, b):
    """Centerline Dice. a, b: boolean vessel masks, same shape."""
    skel_a, skel_b = skeletonize(a), skeletonize(b)
    t_prec = (skel_b & a).sum() / max(skel_b.sum(), 1)
    t_sens = (skel_a & b).sum() / max(skel_a.sum(), 1)
    cldice = 2 * t_prec * t_sens / max(t_prec + t_sens, 1e-9)
    return cldice, t_prec, t_sens


def dilated_iou(a, b, radius):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    a_dil = cv2.dilate(a.astype(np.uint8), k) > 0
    b_dil = cv2.dilate(b.astype(np.uint8), k) > 0
    inter = (a & b_dil) | (b & a_dil)
    return inter.sum() / max((a | b).sum(), 1)


def branch_points(mask):
    skel = skeletonize(mask).astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nb = cv2.filter2D(skel, -1, k, borderType=cv2.BORDER_CONSTANT)
    return int(((skel == 1) & (nb >= 3)).sum())


def noise_floor_perturb(img_rgb):
    """The ONE perturbation used for the noise floor -- nothing else."""
    return cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=0.6)


def vae_roundtrip(img_rgb, vae=None):
    """Encode -> decode through the VAE only, no UNet, no denoising. Requires
    diffusers' AutoencoderKL loaded from sd2-community/stable-diffusion-2-1-base,
    subfolder='vae' -- same checkpoint reference Harriet's pipeline uses."""
    import torch
    t = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        latent = vae.encode(t).latent_dist.sample() * vae.config.scaling_factor
        recon = vae.decode(latent / vae.config.scaling_factor).sample
    recon = ((recon[0].permute(1, 2, 0).clamp(-1, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    return recon


def run_on_folder(src_dir, weights_path, names=None, load_vae=True):
    """
    src_dir: folder containing {name}_original.png and {name}_reconstructed.png
             (raw files -- NOT figure crops).
    weights_path: path to Model A's .h5 weights file -- pass explicitly, do not
                  rely on a hardcoded default (local-machine path and pod path
                  are different, and the pod path's exact filename is unconfirmed).
    Returns a list of dicts, one per image, with clDice/precision/sensitivity vs.
    original, dilated IoU at 2/3/5px, branch points, VAE ceiling clDice, and
    noise-floor clDice.
    """
    names = names or ALL_NAMES
    src_dir = Path(src_dir)

    segmentation.load_model(weights_path)

    vae = None
    if load_vae:
        import torch
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("sd2-community/stable-diffusion-2-1-base", subfolder="vae")
        vae.eval()

    rows = []
    for name in names:
        orig = load_rgb(src_dir / f"{name}_original.png")
        recon = load_rgb(src_dir / f"{name}_reconstructed.png")

        v_orig = vessel_mask(orig)
        v_recon = vessel_mask(recon)

        cldice, prec, sens = cl_dice(v_orig, v_recon)
        row = dict(
            name=name, kind="healthy" if name in HEALTHY else "DR",
            clDice=round(cldice, 3), precision=round(prec, 3), sensitivity=round(sens, 3),
            dilated_IoU_2px=round(dilated_iou(v_orig, v_recon, 2), 3),
            dilated_IoU_3px=round(dilated_iou(v_orig, v_recon, 3), 3),
            dilated_IoU_5px=round(dilated_iou(v_orig, v_recon, 5), 3),
            branch_orig=branch_points(v_orig), branch_recon=branch_points(v_recon),
        )

        if vae is not None:
            vae_out = vae_roundtrip(orig, vae)
            v_vae = vessel_mask(vae_out)
            ceiling_cldice, ceiling_prec, ceiling_sens = cl_dice(v_orig, v_vae)
            row["clDice_ceiling"] = round(ceiling_cldice, 3)
            row["clDice_ceiling_2px"] = round(cl_dice(
                v_orig, cv2.dilate(v_vae.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0)[0], 3)

            perturbed = noise_floor_perturb(orig)
            v_perturbed = vessel_mask(perturbed)
            floor_cldice, _, _ = cl_dice(v_orig, v_perturbed)
            row["clDice_noise_floor"] = round(floor_cldice, 3)

        rows.append(row)

    return rows


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="Folder with {name}_original.png / {name}_reconstructed.png")
    parser.add_argument("--weights", required=True, help="Path to Model A .h5 weights")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    parser.add_argument("--no-vae", action="store_true", help="Skip VAE ceiling/noise-floor (faster)")
    args = parser.parse_args()

    results = run_on_folder(args.src_dir, args.weights, load_vae=not args.no_vae)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    print("Mean clDice (all 13):", round(df["clDice"].mean(), 3))
    print("Mean clDice (DR only):", round(df[df["kind"] == "DR"]["clDice"].mean(), 3))
    print("Mean clDice (healthy only):", round(df[df["kind"] == "healthy"]["clDice"].mean(), 3))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
