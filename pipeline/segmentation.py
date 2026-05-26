"""
pipeline/segmentation.py
Model A — Attention U-Net vessel segmentation (arkanivasarkar).
Loads Keras .h5 weights, runs inference, returns binary vessel mask.
"""

import numpy as np
import cv2


_model = None  # module-level singleton — load once, reuse


def load_model(weights_path: str):
    """Load Attention U-Net from .h5 weights. Call once at startup."""
    global _model
    import tensorflow as tf
    from tensorflow import keras
    _model = keras.models.load_model(weights_path, compile=False)
    print(f'✅ Segmentation model loaded')
    print(f'   Input:  {_model.input_shape}')
    print(f'   Output: {_model.output_shape}')
    return _model


def predict(preprocessed: dict, threshold: float = 0.5) -> np.ndarray:
    """
    Run vessel segmentation on a preprocessed image dict.

    Args:
        preprocessed: output of pipeline.preprocessing.preprocess()
        threshold:    sigmoid threshold for binarisation

    Returns:
        Binary mask (H, W) uint8, 255 = vessel, 0 = background
    """
    if _model is None:
        raise RuntimeError('Model not loaded. Call segmentation.load_model() first.')

    inp_shape = _model.input_shape
    n_channels = inp_shape[-1]

    if n_channels == 1:
        img = preprocessed['green_clahe'].astype(np.float32) / 255.0
        img = img[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)
    else:
        img = preprocessed['enhanced_rgb'].astype(np.float32) / 255.0
        img = img[np.newaxis]                    # (1, H, W, 3)

    pred = _model.predict(img, verbose=0)        # (1, H, W, 1)
    mask = (pred[0, ..., 0] > threshold).astype(np.uint8) * 255
    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological dilation of vessel mask.
    Kernel size 5 → ~10% coverage (supervisor target: 8-12%).
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, k)
