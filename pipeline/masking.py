"""
pipeline/masking.py
Combines vessel mask + lesion exclusion mask into final inpainting mask.
"""

import cv2
import numpy as np


def build_inpaint_mask(vessel_mask: np.ndarray,
                       lesion_mask: np.ndarray,
                       vessel_dilation_kernel: int = 5,
                       lesion_dilation_kernel: int = 3) -> dict:
    """
    Build the final inpainting mask.

    Logic:
      1. Dilate vessel mask (captures perivasculature, hits 8-12% coverage target)
      2. Dilate lesion mask (buffer around lesion edges)
      3. Final = dilated vessels - dilated lesions

    Args:
        vessel_mask:              binary (H, W) uint8 from segmentation
        lesion_mask:              binary (H, W) uint8 from pathology detection
        vessel_dilation_kernel:   px — 5 gives ~10% coverage
        lesion_dilation_kernel:   px — 3 gives tight buffer

    Returns dict:
        vessel_mask:    original undilated vessel mask
        vessel_dilated: dilated vessel mask
        lesion_dilated: dilated lesion mask
        inpaint_mask:   final mask to send to LaMa
        stats:          coverage percentages
    """
    total_px = vessel_mask.size

    # Dilate vessel mask
    k_vessel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (vessel_dilation_kernel, vessel_dilation_kernel)
    )
    vessel_dilated = cv2.dilate(vessel_mask, k_vessel)

    # Dilate lesion mask
    k_lesion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (lesion_dilation_kernel, lesion_dilation_kernel)
    )
    lesion_dilated = cv2.dilate(lesion_mask, k_lesion)

    # Final inpaint mask = vessels - lesions
    inpaint_mask = cv2.bitwise_and(
        vessel_dilated,
        cv2.bitwise_not(lesion_dilated)
    )

    # Stats
    vessel_px         = (vessel_mask > 0).sum()
    vessel_dilated_px = (vessel_dilated > 0).sum()
    lesion_px         = (lesion_dilated > 0).sum()
    inpaint_px        = (inpaint_mask > 0).sum()
    lesion_protection = (vessel_dilated_px - inpaint_px) / max(vessel_dilated_px, 1) * 100

    stats = {
        'vessel_pct':         round(vessel_px / total_px * 100, 2),
        'vessel_dilated_pct': round(vessel_dilated_px / total_px * 100, 2),
        'lesion_pct':         round(lesion_px / total_px * 100, 2),
        'inpaint_pct':        round(inpaint_px / total_px * 100, 2),
        'lesion_protection':  round(lesion_protection, 2),
    }

    return {
        'vessel_mask':    vessel_mask,
        'vessel_dilated': vessel_dilated,
        'lesion_dilated': lesion_dilated,
        'inpaint_mask':   inpaint_mask,
        'stats':          stats,
    }
