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
    """Detect and dilate optic disc region for exclusion from lesion detectors."""
    l = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB))[0]
    _, od = cv2.threshold(l, 220, 255, cv2.THRESH_BINARY)
    od = cv2.dilate(od, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
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
                         percentile: int = 97,
                         min_brightness: int = 180,
                         min_area: int = 10) -> np.ndarray:
    """
    Bright yellow/white waxy deposits.
    97th-percentile L threshold adapts to each image's brightness distribution.
    """
    l = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB))[0]
    fov_pixels = l[(od_mask == 0) & (l > 10)]
    thresh = int(np.percentile(fov_pixels, percentile)) if len(fov_pixels) > 0 else 200
    thresh = max(thresh, min_brightness)

    _, mask = cv2.threshold(l, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(od_mask))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Remove speckles
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
                          blackhat_kernel: int = 21,
                          threshold: int = 20,
                          min_area: int = 8,
                          max_area: int = 50,
                          min_circularity: float = 0.45,
                          max_eccentricity: float = 0.80) -> np.ndarray:
    """
    Tiny dark red dots. Black top-hat morphology + shape filter.
    """
    green = img_rgb[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_kernel, blackhat_kernel))
    tophat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
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
