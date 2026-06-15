"""
pipeline/pathology.py
Rule-based pathology detection for DR lesions.
Detects: hard exudates, haemorrhages, microaneurysms.
Returns exclusion masks — lesion regions to protect from inpainting.
"""

import cv2
import numpy as np
from skimage import measure as sk_measure


def detect_optic_disc(img_rgb: np.ndarray) -> np.ndarray:
    """Detect and dilate optic disc region for exclusion from lesion detectors.
    Keeps only the largest bright connected component so exudates (small scattered
    bright patches) don't get absorbed into the exclusion mask."""
    l = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB))[0]
    _, od = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY)
    # Keep only the largest blob — the disc is one large region; exudates are scattered small ones
    labeled = sk_measure.label(od > 0)
    if labeled.max() > 0:
        regions = sk_measure.regionprops(labeled)
        largest = max(regions, key=lambda r: r.area)
        od = (labeled == largest.label).astype(np.uint8) * 255
    od = cv2.dilate(od, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200)))
    return od


def _filter_by_shape(mask: np.ndarray,
                     min_circ: float = 0.4,
                     max_ecc: float = 0.85) -> np.ndarray:
    """Keep only blobs matching circularity + eccentricity criteria."""
    out = np.zeros_like(mask)
    labeled = sk_measure.label(mask > 0)
    for region in sk_measure.regionprops(labeled):
        if region.perimeter == 0:
            continue
        circ = (4 * np.pi * region.area) / (region.perimeter ** 2)
        if circ >= min_circ and region.eccentricity <= max_ecc:
            out[labeled == region.label] = 255
    return out


def detect_hard_exudates(img_rgb: np.ndarray,
                         od_mask: np.ndarray,
                         tophat_kernel: int = 45,
                         threshold: int = 35,
                         min_area: int = 10) -> np.ndarray:
    """
    Bright yellow/white waxy deposits.
    White top-hat on green channel — highlights bright blobs relative to local background.
    Works at full resolution; kernel sized for full-res images (2848x4288).
    """
    green = img_rgb[:, :, 1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel, tophat_kernel))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, k)
    _, mask = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(od_mask))

    labeled = sk_measure.label(mask > 0)
    filtered = np.zeros_like(mask)
    for region in sk_measure.regionprops(labeled):
        if region.area >= min_area:
            filtered[labeled == region.label] = 255
    return filtered


def detect_haemorrhages(img_rgb: np.ndarray,
                        od_mask: np.ndarray,
                        green_threshold: int = 80,
                        min_area: int = 30,
                        max_area: int = 2000,
                        min_circularity: float = 0.4,
                        max_eccentricity: float = 0.85) -> np.ndarray:
    """
    Dark red blobs. High R/low G. Shape filter removes vessel fragments.
    """
    green = img_rgb[:, :, 1]
    _, dark = cv2.threshold(green, green_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(dark, cv2.bitwise_not(od_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    labeled = sk_measure.label(mask > 0)
    sized = np.zeros_like(mask)
    for region in sk_measure.regionprops(labeled):
        if min_area <= region.area <= max_area:
            sized[labeled == region.label] = 255

    return _filter_by_shape(sized, min_circ=min_circularity, max_ecc=max_eccentricity)


def detect_microaneurysms(img_rgb: np.ndarray,
                          od_mask: np.ndarray,
                          threshold: int = 12,
                          min_area: int = 20,
                          max_area: int = 1500,
                          min_circularity: float = 0.1,
                          max_eccentricity: float = 0.99) -> np.ndarray:
    """
    Shade correction approach: subtract large median-filtered background from
    Gaussian-smoothed green channel. Removes slow background variation (vessels,
    illumination gradients) and leaves small dark spots (MAs).
    Key step from literature — this is what separates good MA detectors from poor ones.
    """
    green = img_rgb[:, :, 1]

    # Gaussian filter enhances small structures and suppresses noise
    smoothed = cv2.GaussianBlur(green, (3, 3), 1.0)

    # Background estimation via large median filter — captures slow intensity variation
    bg = cv2.medianBlur(green, 25)

    # Shade correction: bg - smoothed > 0 where pixel is darker than background (i.e. MA site)
    shade_corr = np.clip(bg.astype(np.int16) - smoothed.astype(np.int16), 0, 255).astype(np.uint8)

    _, mask = cv2.threshold(shade_corr, threshold, 255, cv2.THRESH_BINARY)

    # Closing merges nearby pixels into coherent blobs (shade correction tends to fragment)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(od_mask))

    labeled = sk_measure.label(mask > 0)
    sized = np.zeros_like(mask)
    for region in sk_measure.regionprops(labeled):
        if min_area <= region.area <= max_area:
            sized[labeled == region.label] = 255

    return _filter_by_shape(sized, min_circ=min_circularity, max_ecc=max_eccentricity)


def detect_all(img_rgb: np.ndarray, cfg: dict = None) -> dict:
    """
    Run all detectors. Returns combined mask + per-class dict.

    Args:
        img_rgb: CLAHE-enhanced RGB image (H, W, 3)
        cfg:     pathology config dict (from default.yaml). Uses defaults if None.

    Returns dict:
        combined      : uint8 (H, W) — union of all lesion masks
        exudates      : uint8 (H, W)
        haemorrhages  : uint8 (H, W)
        microaneurysms: uint8 (H, W)
        optic_disc    : uint8 (H, W)
    """
    cfg = cfg or {}
    ex_cfg = cfg.get('hard_exudate', {})
    ha_cfg = cfg.get('haemorrhage', {})
    ma_cfg = cfg.get('microaneurysm', {})

    od = detect_optic_disc(img_rgb)
    ex = detect_hard_exudates(img_rgb, od, **ex_cfg)
    ha = detect_haemorrhages(img_rgb, od, **ha_cfg)
    ma = detect_microaneurysms(img_rgb, od, **ma_cfg)

    combined = cv2.bitwise_or(cv2.bitwise_or(ex, ha), ma)

    return {
        'combined':       combined,
        'exudates':       ex,
        'haemorrhages':   ha,
        'microaneurysms': ma,
        'optic_disc':     od,
    }
