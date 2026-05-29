"""
pipeline/inpainting.py
Stable Diffusion Inpainting wrapper (Path A — Phase 1).
Replaces LaMa for vessel mask inpainting.

Phase 1: vanilla SD inpainting, no ControlNet.
Phase 2 (ControlNet) can be added by extending load_model() and inpaint()
with a controlnet_image argument — the rest of the pipeline is unchanged.

Model and all inference params are driven from configs/default.yaml
under the `inpainting:` key.
"""

import numpy as np
import cv2
from PIL import Image


_sd_pipe = None


def load_model(cfg: dict = None, device: str = 'cuda'):
    """
    Load SD inpainting pipeline. Call once at startup.

    Args:
        cfg:    inpainting config dict (from default.yaml `inpainting:` block)
        device: cuda | cpu
    """
    global _sd_pipe

    cfg = cfg or {}
    model_id = cfg.get('sd_model', 'runwayml/stable-diffusion-inpainting')

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    print(f'Loading SD inpainting model: {model_id} ...')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,   # safe to disable for medical content
    ).to(device)

    # Slight memory saving with no quality loss
    pipe.enable_attention_slicing()

    _sd_pipe = {
        'pipe':   pipe,
        'device': device,
        'cfg':    cfg,
    }

    print(f'✅ SD inpainting loaded (device={device})')
    return _sd_pipe


def inpaint(image_rgb: np.ndarray,
            mask: np.ndarray,
            device: str = 'cuda',
            seed: int = None) -> np.ndarray:
    """
    Run SD inpainting on a single image.

    Args:
        image_rgb: uint8 (H, W, 3) RGB — original fundus image
        mask:      uint8 (H, W)    — inpaint mask (255 = fill, 0 = keep)
        device:    cuda | cpu
        seed:      random seed for reproducibility.
                   None = random per image (recommended for privacy —
                   different seed = different synthetic vessels).

    Returns:
        uint8 (H, W, 3) RGB — de-identified image
    """
    if _sd_pipe is None:
        raise RuntimeError('SD pipeline not loaded. Call inpainting.load_model() first.')

    import torch

    pipe = _sd_pipe['pipe']
    cfg  = _sd_pipe['cfg']

    prompt          = cfg.get('prompt', '').strip()
    negative_prompt = cfg.get('negative_prompt', '').strip()
    steps           = int(cfg.get('num_inference_steps', 50))
    guidance_scale  = float(cfg.get('guidance_scale', 7.5))
    strength        = float(cfg.get('strength', 1.0))

    # SD expects PIL images at 512×512
    h, w = image_rgb.shape[:2]
    pil_image = Image.fromarray(image_rgb).resize((512, 512))
    pil_mask  = Image.fromarray(mask).resize((512, 512), resample=Image.NEAREST)

    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())

    generator = torch.Generator(device=device).manual_seed(seed)

    result_pil = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        mask_image=pil_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    ).images[0]

    # Resize back to original resolution and return as numpy uint8
    result_np = np.array(result_pil.resize((w, h), resample=Image.LANCZOS))
    return result_np.astype(np.uint8)


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
    from pathlib import Path
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

        mask = mask_dict[stem]
        result = inpaint(img_rgb, mask, device=device)  # seed=None → random per image

        out_path = output_dir / f'{stem}_deid.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        output_paths.append(out_path)

    return output_paths
