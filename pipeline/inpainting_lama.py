"""
pipeline/inpainting.py
LaMa inpainting wrapper.
Fills vessel mask regions with plausible retinal background texture.
"""

import cv2
import numpy as np
from pathlib import Path


_lama_model = None


def load_model(weights_dir: str, device: str = 'cuda'):
    """Load LaMa model. Call once at startup."""
    global _lama_model
    try:
        from lama_cleaner.model.lama import LaMa
        from lama_cleaner.schema import Config as LamaConfig
        _lama_model = {
            'model': LaMa(device=device),
            'device': device,
            'backend': 'lama_cleaner',
        }
        print(f'✅ LaMa loaded via lama-cleaner (device={device})')
    except ImportError:
        # Fallback: try loading LaMa directly from big-lama repo
        import torch
        import sys
        sys.path.insert(0, str(Path(weights_dir).parent))
        _lama_model = {
            'weights_dir': weights_dir,
            'device': device,
            'backend': 'direct',
        }
        print(f'✅ LaMa loaded directly (device={device})')
    return _lama_model


def inpaint(image_rgb: np.ndarray,
            mask: np.ndarray,
            device: str = 'cuda') -> np.ndarray:
    """
    Run LaMa inpainting on a single image.

    Args:
        image_rgb: uint8 (H, W, 3) — original image
        mask:      uint8 (H, W) — inpaint mask (255 = fill, 0 = keep)
        device:    cuda | cpu

    Returns:
        uint8 (H, W, 3) — de-identified image
    """
    if _lama_model is None:
        raise RuntimeError('LaMa not loaded. Call inpainting.load_model() first.')

    if _lama_model['backend'] == 'lama_cleaner':
        from lama_cleaner.schema import Config as LamaConfig
        model = _lama_model['model']
        cfg = LamaConfig(
            ldm_steps=20,
            ldm_sampler='plms',
            zits_wireframe=False,
            hd_strategy='Resize',
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=512,
            hd_strategy_resize_limit=512,
        )
        result = model(image_rgb, mask, cfg)
        # Ensure the output is uint8 for OpenCV compatibility
        if np.issubdtype(result.dtype, np.floating):
            if result.max() <= 1.0:
                result = result * 255.0
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = result.astype(np.uint8)
        return result

    elif _lama_model['backend'] == 'direct':
        # Direct LaMa call — handles the model loading and inference
        import torch
        from PIL import Image
        import torchvision.transforms as T

        weights_dir = _lama_model['weights_dir']

        # Convert to tensors
        img_tensor = T.ToTensor()(Image.fromarray(image_rgb)).unsqueeze(0)
        mask_tensor = T.ToTensor()(Image.fromarray(mask)).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        img_tensor  = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        # Masked input to model
        masked = img_tensor * (1 - mask_tensor)

        with torch.no_grad():
            # Load model if needed
            from omegaconf import OmegaConf
            from saicinpainting.training.trainers import load_checkpoint

            train_config_path = Path(weights_dir) / 'config.yaml'
            cfg = OmegaConf.load(train_config_path)
            cfg.training_model.predict_only = True
            cfg.visualizer.kind = 'noop'

            checkpoint_path = Path(weights_dir) / 'models' / 'best.ckpt'
            model = load_checkpoint(cfg, str(checkpoint_path), strict=False, map_location=device)
            model.eval()
            model.to(device)

            batch = {'image': masked, 'mask': mask_tensor}
            result_batch = model(batch)
            result = result_batch['inpainted'][0].permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)

        return result


def inpaint_batch(image_paths: list,
                  mask_dict: dict,
                  output_dir: str,
                  device: str = 'cuda') -> list:
    """
    Batch inpainting over a list of image paths.

    Args:
        image_paths: list of Path objects
        mask_dict:   {stem: inpaint_mask np.ndarray}
        output_dir:  where to save de-identified images
        device:      cuda | cpu

    Returns:
        list of output paths
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for path in tqdm(image_paths, desc='Inpainting'):
        stem = path.stem
        if stem not in mask_dict:
            print(f'  ⚠️  No mask for {stem}, skipping')
            continue

        img = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (512, 512))

        mask = mask_dict[stem]
        result = inpaint(img_rgb, mask, device=device)

        out_path = output_dir / f'{stem}_deid.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        output_paths.append(out_path)

    return output_paths
