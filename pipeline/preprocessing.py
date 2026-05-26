"""
pipeline/preprocessing.py
FOV detection, CLAHE enhancement, histogram normalisation.
Everything that happens before segmentation.
"""

import cv2
import numpy as np
import skimage.exposure as skie
from pathlib import Path


def detect_fov(img_rgb: np.ndarray) -> tuple[int, int, int]:
    """
    Detect the circular retinal field-of-view in a fundus image.
    Handles off-centre EyePACS images.
    Returns (cx, cy, r).
    """
    green = img_rgb[:, :, 1]
    blur = cv2.GaussianBlur(green, (31, 31), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = img_rgb.shape[:2]
        return w // 2, h // 2, min(w, h) // 2 - 10
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest)
    return int(cx), int(cy), int(r) - 5


def apply_fov_mask(img_np: np.ndarray,
                   cx: int = None,
                   cy: int = None,
                   r: int = None) -> np.ndarray:
    """Zero out pixels outside the circular retinal field."""
    h, w = img_np.shape[:2]
    if cx is None:
        cx, cy = w // 2, h // 2
        r = min(cx, cy) - 10
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    if img_np.ndim == 3:
        mask = mask[:, :, np.newaxis]
    return (img_np * (mask / 255)).astype(img_np.dtype)


def preprocess(img_path: str | Path,
               target_size: int = 512,
               clahe_clip: float = 2.0,
               clahe_tile: tuple = (8, 8),
               reference_rgb: np.ndarray = None) -> dict:
    """
    Full preprocessing pipeline for a single fundus image.

    Steps:
      1. Load + resize
      2. Detect + apply FOV mask
      3. CLAHE on L channel
      4. Optional: histogram match to reference (normalisation)

    Returns dict:
      original_rgb  : uint8 (H, W, 3) — resized + FOV masked
      enhanced_rgb  : uint8 (H, W, 3) — CLAHE enhanced + FOV masked
      green_clahe   : uint8 (H, W)    — green channel CLAHE
      normalised_rgb: uint8 (H, W, 3) — hist-matched (if reference given)
      fov           : (cx, cy, r)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'Cannot read: {img_path}')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (target_size, target_size))

    cx, cy, r = detect_fov(img_rgb)
    img_rgb_masked = apply_fov_mask(img_rgb, cx, cy, r)

    # CLAHE on L channel
    clahe_op = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    lab = cv2.cvtColor(img_rgb_masked, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe_op.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2RGB)
    green_clahe = clahe_op.apply(img_rgb_masked[:, :, 1])

    result = {
        'original_rgb': img_rgb_masked,
        'enhanced_rgb': enhanced,
        'green_clahe':  green_clahe,
        'fov':          (cx, cy, r),
    }

    # Optional histogram normalisation
    if reference_rgb is not None:
        ref_l = cv2.split(cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB))[0]
        l_matched = np.clip(
            skie.match_histograms(l_clahe, ref_l), 0, 255
        ).astype(np.uint8)
        norm_rgb = cv2.cvtColor(cv2.merge([l_matched, a, b]), cv2.COLOR_LAB2RGB)
        orig_dark = img_rgb.sum(axis=-1) < 30
        norm_rgb[orig_dark] = 0
        result['normalised_rgb'] = apply_fov_mask(norm_rgb, cx, cy, r)

    return result


def select_reference(image_paths: list, n_candidates: int = 30) -> np.ndarray:
    """Pick the highest-contrast image from a random sample as histogram reference."""
    import random
    candidates = random.sample(image_paths, min(n_candidates, len(image_paths)))
    best_std, best_img = -1, None
    for p in candidates:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (512, 512))
        std = np.std(img_rgb[:, :, 1])
        if std > best_std:
            best_std = std
            best_img = img_rgb
    return best_img
