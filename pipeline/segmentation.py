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
    
    # Disable GPU for TensorFlow to bypass CuDNN version mismatch
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("✅ Disabled GPU for TensorFlow (segmentation will run on CPU)")
    except Exception as e:
        print(f"Note: Could not disable GPU for TensorFlow: {e}")

    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras

    import sys
    from pathlib import Path

    # Try reconstructing model architecture and loading weights (to avoid bytecode compatibility issues)
    try:
        candidate_paths = [
            '/workspace/arkan_unet',
            '/workspace/realizemd_deid/arkan_unet',
            str(Path(__file__).parent.parent.parent / 'arkan_unet'),
        ]
        for p in candidate_paths:
            path_obj = Path(p).resolve()
            if path_obj.exists():
                if str(path_obj) not in sys.path:
                    sys.path.insert(0, str(path_obj))
                break

        from model import attentionunet
        with tf.device('/cpu:0'):
            _model = attentionunet(input_shape=(512, 512, 1))
            _model.load_weights(weights_path)
        print(f'✅ Segmentation model loaded by reconstructing architecture')
    except Exception as e:
        print(f"⚠️  Could not reconstruct model architecture: {e}. Falling back to load_model...")
        # Monkeypatch Conv2DTranspose to ignore 'groups' (Keras 3 compatibility workaround)
        try:
            Conv2DTranspose = keras.layers.Conv2DTranspose
            original_init = Conv2DTranspose.__init__
            def patched_init(self, *args, **kwargs):
                kwargs.pop('groups', None)
                return original_init(self, *args, **kwargs)
            Conv2DTranspose.__init__ = patched_init
        except Exception as patch_err:
            print(f"Note: Could not patch Conv2DTranspose: {patch_err}")

        # Enable unsafe deserialization for Lambda layers in Keras 3
        if hasattr(keras, 'config') and hasattr(keras.config, 'enable_unsafe_deserialization'):
            keras.config.enable_unsafe_deserialization()

        try:
            with tf.device('/cpu:0'):
                _model = keras.models.load_model(weights_path, compile=False, safe_mode=False)
        except TypeError:
            with tf.device('/cpu:0'):
                _model = keras.models.load_model(weights_path, compile=False)
        print(f'✅ Segmentation model loaded via load_model')
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

    import tensorflow as tf

    inp_shape = _model.input_shape
    n_channels = inp_shape[-1]

    if n_channels == 1:
        img = preprocessed['green_clahe'].astype(np.float32) / 255.0
        img = img[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)
    else:
        img = preprocessed['enhanced_rgb'].astype(np.float32) / 255.0
        img = img[np.newaxis]                    # (1, H, W, 3)

    with tf.device('/cpu:0'):
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
