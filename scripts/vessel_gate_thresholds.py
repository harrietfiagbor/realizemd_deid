"""
vessel_gate_thresholds.py

Task 3: set the tolerance-based vessel gate thresholds Adam asked for (he withdrew
the >=0.80 exact-clDice target because a 1px shift alone drops exact clDice to
~0.60). The gate is: tolerance-based clDice (2-3px) + caliber-stratified recovery,
with thresholds set FROM THE VAE CEILING under the tolerant metric (the physical
best case -- even a perfect diffusion model can't beat what the VAE preserves).

For each image this computes, against the Model-A vessel mask of the ORIGINAL:
  - CEILING  = VAE-only round-trip (encode->decode, no UNet)
  - CURRENT  = the conditioned reconstruction (where we actually are now)
for two metrics:
  - tolerant clDice at 2px and 3px
  - caliber-stratified recovery (thick >=5px dia / mid 3-5px / thin <3px), 2px tol

Then proposes gate thresholds as a fraction of the ceiling, and prints where
CURRENT sits so the headroom is visible.

Built-in sanity checks (printed, and gate numbers suppressed if they fail):
  1. tolerant clDice >= exact clDice for every image (by construction).
  2. 3px clDice >= 2px clDice for every image (monotone in tolerance).
  3. ceiling >= current for the tolerant clDice (VAE should beat the conditioned
     recon -- if not, something is off in the conditioned files or the pipeline).

Shared metric functions imported from vessel_metrics_reference.py (the script of
record) so there is NO second implementation to drift.

RunPod usage:
    python scripts/vessel_gate_thresholds.py \
        /workspace/diffusion_retina/conditioned_reconstructions \
        --weights "/workspace/models/attention_unet/<actual filename>.h5" \
        --out /workspace/diffusion_retina_victoria/vessel_gate_thresholds.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so this file's dir (scripts/) is importable
from vessel_metrics_reference import (
    load_rgb, vessel_mask, vae_roundtrip, cl_dice, cl_dice_tol, HEALTHY, DR, ALL_NAMES,
)
from pipeline import segmentation

import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

CALIBER_BANDS = [("thick(>=5px)", 2.5, 99.0), ("mid(3-5px)", 1.5, 2.5), ("thin(<3px)", 0.0, 1.5)]


def caliber_recovery(v_orig, v_other, tol=2):
    """Fraction of the ORIGINAL centerline (per caliber band, by local radius)
    recovered within `tol` px in v_other. Same implementation as the notebook."""
    dist = ndi.distance_transform_edt(v_orig)
    skel = skeletonize(v_orig)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    other_tol = cv2.dilate(v_other.astype(np.uint8), k) > 0
    out = {}
    for label, lo, hi in CALIBER_BANDS:
        band = skel & (dist >= lo) & (dist < hi)
        n = int(band.sum())
        out[label] = (float((band & other_tol).sum()) / max(n, 1), n)
    return out


def main():
    import argparse
    import pandas as pd
    from diffusers import AutoencoderKL

    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    segmentation.load_model(args.weights)
    vae = AutoencoderKL.from_pretrained("sd2-community/stable-diffusion-2-1-base", subfolder="vae"); vae.eval()

    src = Path(args.src_dir)
    rows = []
    sanity_fail = []
    for name in ALL_NAMES:
        orig = load_rgb(src / f"{name}_original.png")
        recon = load_rgb(src / f"{name}_reconstructed.png")
        v_orig = vessel_mask(orig)
        v_recon = vessel_mask(recon)
        v_vae = vessel_mask(vae_roundtrip(orig, vae))

        ex_cur = cl_dice(v_orig, v_recon)[0]
        t2_cur = cl_dice_tol(v_orig, v_recon, 2)[0]
        t3_cur = cl_dice_tol(v_orig, v_recon, 3)[0]
        t2_ceil = cl_dice_tol(v_orig, v_vae, 2)[0]
        t3_ceil = cl_dice_tol(v_orig, v_vae, 3)[0]

        # sanity
        if not (t2_cur >= ex_cur - 1e-6):           sanity_fail.append(f"{name}: 2px<exact (current)")
        if not (t3_cur >= t2_cur - 1e-6):           sanity_fail.append(f"{name}: 3px<2px (current)")
        if not (t2_ceil >= t2_cur - 1e-6):          sanity_fail.append(f"{name}: ceiling<current (2px)")

        cal_cur = caliber_recovery(v_orig, v_recon, tol=2)
        cal_ceil = caliber_recovery(v_orig, v_vae, tol=2)

        row = dict(
            name=name, kind="healthy" if name in HEALTHY else "DR",
            clDice_exact_cur=round(ex_cur, 3),
            clDice_2px_cur=round(t2_cur, 3), clDice_2px_ceil=round(t2_ceil, 3),
            clDice_3px_cur=round(t3_cur, 3), clDice_3px_ceil=round(t3_ceil, 3),
        )
        for label, _, _ in CALIBER_BANDS:
            row[f"{label}_cur"] = round(100 * cal_cur[label][0], 1)
            row[f"{label}_ceil"] = round(100 * cal_ceil[label][0], 1)
        rows.append(row)
        print(f"{name:13s} 2px cur={t2_cur:.3f} ceil={t2_ceil:.3f} | "
              f"thin cur={100*cal_cur['thin(<3px)'][0]:.0f}% ceil={100*cal_ceil['thin(<3px)'][0]:.0f}%")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    if sanity_fail:
        print("SANITY CHECKS FAILED -- thresholds NOT emitted, fix the implementation:")
        for f in sanity_fail:
            print("  -", f)
    else:
        print("Sanity checks PASS (tolerant>=exact, 3px>=2px, ceiling>=current).\n")
        for grp, sub in [("DR", df[df.kind == "DR"]), ("healthy", df[df.kind == "healthy"]), ("ALL", df)]:
            print(f"[{grp}] n={len(sub)}")
            print(f"   2px clDice: ceiling mean={sub.clDice_2px_ceil.mean():.3f} (min {sub.clDice_2px_ceil.min():.3f}) | "
                  f"current mean={sub.clDice_2px_cur.mean():.3f}")
            print(f"   3px clDice: ceiling mean={sub.clDice_3px_ceil.mean():.3f} (min {sub.clDice_3px_ceil.min():.3f}) | "
                  f"current mean={sub.clDice_3px_cur.mean():.3f}")
            print(f"   thin-vessel recovery: ceiling mean={sub['thin(<3px)_ceil'].mean():.1f}% | "
                  f"current mean={sub['thin(<3px)_cur'].mean():.1f}%")
            print()

        ceil_2px_all = df.clDice_2px_ceil.mean()
        thin_ceil_all = df['thin(<3px)_ceil'].mean()
        print("PROPOSED GATE (set at 90% of the tolerant VAE ceiling):")
        print(f"   - 2px tolerant clDice  >= {0.90 * ceil_2px_all:.3f}  (90% of ceiling mean {ceil_2px_all:.3f})")
        print(f"   - thin-vessel recovery >= {0.90 * thin_ceil_all:.0f}% (90% of ceiling mean {thin_ceil_all:.1f}%)")
        print("   (thick/mid vessels already recover ~88-97%; thin is the binding constraint,")
        print("    so it's the band the gate should actually turn on.)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
