"""
pipeline/deidentify.py
Single public interface: deidentify(image) -> image
This is the function RealizeMD will call in production.
"""

import numpy as np
import cv2
from pathlib import Path

from . import preprocessing, segmentation, pathology, masking, inpainting


def deidentify(image_rgb: np.ndarray,
               img_path: str | Path = None,
               reference_rgb: np.ndarray = None,
               cfg: dict = None,
               return_intermediates: bool = False) -> np.ndarray | dict:
    """
    De-identify a single fundus image by inpainting vessel patterns.

    Pipeline:
      1. Preprocess (FOV mask, CLAHE)
      2. Segment vessels (Model A)
      3. Detect pathology (rule-based)
      4. Build inpaint mask (vessels - lesions, dilated)
      5. SD inpainting

    Args:
        image_rgb:           uint8 (H, W, 3) RGB image
            OR pass img_path to load from disk
        reference_rgb:       optional reference for histogram normalisation
        cfg:                 config dict (from default.yaml). Uses defaults if None.
        return_intermediates: if True, return full dict with all intermediate masks

    Returns:
        uint8 (H, W, 3) de-identified image
        OR dict with 'deid_image' + all intermediates if return_intermediates=True
    """
    cfg = cfg or {}
    pp_cfg   = cfg.get('preprocessing', {})
    seg_cfg  = cfg.get('segmentation', {})
    mask_cfg = cfg.get('vessel_mask', {})
    path_cfg = cfg.get('pathology', {})
    inp_cfg  = cfg.get('inpainting', {})

    # ── 1. Load + preprocess ─────────────────────────────────────────────────
    if img_path is not None:
        preprocessed = preprocessing.preprocess(
            img_path,
            target_size=pp_cfg.get('target_size', 512),
            clahe_clip=pp_cfg.get('clahe_clip', 2.0),
            clahe_tile=tuple(pp_cfg.get('clahe_tile', [8, 8])),
            reference_rgb=reference_rgb,
        )
    else:
        # image_rgb passed directly — resize + apply FOV
        target_size = pp_cfg.get('target_size', 512)
        img_resized = cv2.resize(image_rgb, (target_size, target_size))
        cx, cy, r = preprocessing.detect_fov(img_resized)
        img_masked = preprocessing.apply_fov_mask(img_resized, cx, cy, r)
        clahe_op = cv2.createCLAHE(
            clipLimit=pp_cfg.get('clahe_clip', 2.0),
            tileGridSize=tuple(pp_cfg.get('clahe_tile', [8, 8]))
        )
        import cv2 as _cv2
        lab = _cv2.cvtColor(img_masked, _cv2.COLOR_RGB2LAB)
        l, a, b = _cv2.split(lab)
        l_clahe = clahe_op.apply(l)
        enhanced = _cv2.cvtColor(_cv2.merge([l_clahe, a, b]), _cv2.COLOR_LAB2RGB)
        preprocessed = {
            'original_rgb': img_masked,
            'enhanced_rgb': enhanced,
            'green_clahe':  clahe_op.apply(img_masked[:, :, 1]),
            'fov':          (cx, cy, r),
        }

    # ── 2. Vessel segmentation ────────────────────────────────────────────────
    vessel_mask = segmentation.predict(
        preprocessed,
        threshold=seg_cfg.get('threshold', 0.5)
    )

    # ── 3. Pathology detection ────────────────────────────────────────────────
    clahe_img = preprocessed.get('normalised_rgb', preprocessed['enhanced_rgb'])
    lesion_result = pathology.detect_all(clahe_img, path_cfg)

    # ── 4. Build inpaint mask ─────────────────────────────────────────────────
    mask_result = masking.build_inpaint_mask(
        vessel_mask=vessel_mask,
        lesion_mask=lesion_result['combined'],
        vessel_dilation_kernel=mask_cfg.get('dilation_kernel', 5),
        lesion_dilation_kernel=path_cfg.get('lesion_dilation_kernel', 3),
    )

    # ── 5. Inpaint ────────────────────────────────────────────────────────────
    deid_image = inpainting.inpaint(
        image_rgb=preprocessed['original_rgb'],
        mask=mask_result['inpaint_mask'],
        device=inp_cfg.get('device', 'cuda'),
    )

    if return_intermediates:
        return {
            'deid_image':      deid_image,
            'original_rgb':    preprocessed['original_rgb'],
            'vessel_mask':     vessel_mask,
            'vessel_dilated':  mask_result['vessel_dilated'],
            'lesion_mask':     lesion_result['combined'],
            'lesion_parts':    lesion_result,
            'inpaint_mask':    mask_result['inpaint_mask'],
            'mask_stats':      mask_result['stats'],
            'fov':             preprocessed['fov'],
        }

    return deid_image


def deidentify_batch(image_paths: list,
                     output_dir: str,
                     reference_rgb: np.ndarray = None,
                     cfg: dict = None) -> list:
    """
    Batch de-identification. Saves outputs to output_dir.
    Returns list of (original_path, deid_path) tuples.
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for path in tqdm(image_paths, desc='De-identifying'):
        try:
            deid = deidentify(
                image_rgb=None,
                img_path=path,
                reference_rgb=reference_rgb,
                cfg=cfg,
            )
            out_path = output_dir / f'{Path(path).stem}_deid.png'
            import cv2 as _cv2
            _cv2.imwrite(str(out_path), _cv2.cvtColor(deid, _cv2.COLOR_RGB2BGR))
            results.append((path, out_path))
        except Exception as e:
            print(f'  ⚠️  Failed {Path(path).name}: {e}')

    return results
