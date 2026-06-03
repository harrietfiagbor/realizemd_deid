"""
pipeline/vessel_pattern.py
Generates a privacy-safe synthetic vessel conditioning pattern for ControlNet.

Option B (from Path C spec): elastic deformation + rotation + flip applied to
the original vessel mask to break topological identity while preserving
anatomical plausibility.

Kept separate from masking.py intentionally — masking builds the inpaint zone,
this builds the ControlNet conditioning signal. Different concerns.

Option A (cross-patient transplant) can be added here later without touching
anything else in the pipeline.
"""

import cv2
import numpy as np
from PIL import Image


def deform_vessel_pattern(
    vessel_mask: np.ndarray, cfg: dict = None, seed: int = None
) -> np.ndarray:
    """
    Apply elastic deformation + rotation + optional flip to a vessel mask
    to produce a topologically distinct conditioning pattern for ControlNet.

    Args:
        vessel_mask: uint8 (H, W) binary vessel mask (255 = vessel)
        cfg:         vessel_pattern config dict (from default.yaml)
        seed:        random seed. Should match the seed used in inpainting.inpaint()
                     so the deformation is reproducible per image.

    Returns:
        uint8 (H, W) deformed binary vessel mask, same shape as input
    """
    cfg = cfg or {}
    rng = np.random.RandomState(seed)

    alpha = cfg.get("elastic_alpha", 200)  # displacement magnitude
    sigma = cfg.get("elastic_sigma", 20)  # smoothness of displacement
    rotate_min = cfg.get("rotate_min", 15)
    rotate_max = cfg.get("rotate_max", 60)
    flip_prob = cfg.get("flip_prob", 0.3)

    h, w = vessel_mask.shape[:2]

    # ── 1. Elastic deformation ────────────────────────────────────────────────
    # Random displacement fields, smoothed by Gaussian
    dx = (
        cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma)
        * alpha
    )
    dy = (
        cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma)
        * alpha
    )

    # Remap coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)

    deformed = cv2.remap(
        vessel_mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    deformed = (deformed > 127).astype(np.uint8) * 255

    # ── 2. Rotation ───────────────────────────────────────────────────────────
    # Forced rotation — breaks radial symmetry that elastic alone can't fully disrupt
    angle = rng.uniform(rotate_min, rotate_max)
    if rng.rand() > 0.5:
        angle = -angle  # randomise direction

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale=1.0)
    deformed = cv2.warpAffine(
        deformed,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # ── 3. Optional horizontal flip ───────────────────────────────────────────
    if rng.rand() < flip_prob:
        deformed = cv2.flip(deformed, 1)

    return deformed


def build_control_image(
    vessel_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    cfg: dict = None,
    seed: int = None,
) -> Image.Image:
    """
    Build the ControlNet conditioning image (PIL RGB, 512×512).

    Steps:
      1. Deform the vessel mask
      2. Restrict the deformed pattern to inside the inpaint region only
         (per spec: don't condition SD on vessel structure outside the fill zone)
      3. Convert to 3-channel PIL Image for diffusers

    Args:
        vessel_mask:  uint8 (H, W) binary vessel mask from segmentation
        inpaint_mask: uint8 (H, W) inpaint mask (255 = fill region)
        cfg:          vessel_pattern config dict
        seed:         random seed

    Returns:
        PIL RGB image to pass as control_image to the ControlNet pipeline
    """
    deformed = deform_vessel_pattern(vessel_mask, cfg=cfg, seed=seed)

    # Restrict to inpaint zone — vessels outside the fill region aren't relevant
    # to the conditioning and could confuse the model
    inpaint_binary = (inpaint_mask > 0).astype(np.uint8)
    deformed_masked = cv2.bitwise_and(deformed, deformed, mask=inpaint_binary)

    # Invert: scribble ControlNet expects black lines on white background
    # Original mask is white vessels on black — flip it
    inverted = cv2.bitwise_not(deformed_masked)
    control_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(control_rgb)
