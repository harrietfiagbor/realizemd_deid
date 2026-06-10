"""
pipeline/inpainting.py
SD + ControlNet Inpainting wrapper (Path C).

Phase 1 (vanilla SD) is retired — see inpainting_lama.py for LaMa and the
original SD-only code is preserved in git history.

Path C: StableDiffusionControlNetInpaintPipeline with lllyasviel/sd-controlnet-scribble.
ControlNet conditioning image = elastically deformed vessel mask (from vessel_pattern.py).

All inference params driven from configs/default.yaml under `inpainting:`.
"""

import numpy as np
import cv2
from PIL import Image


_sd_pipe = None


def load_model(cfg: dict = None, device: str = 'cuda'):
    """
    Load SD + ControlNet inpainting pipeline. Call once at startup.

    Args:
        cfg:    inpainting config dict (from default.yaml `inpainting:` block)
        device: cuda | cpu
    """
    global _sd_pipe

    cfg = cfg or {}
    sd_model         = cfg.get('sd_model',          'runwayml/stable-diffusion-inpainting')
    controlnet_model = cfg.get('controlnet_model',  'lllyasviel/sd-controlnet-scribble')

    import torch
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

    print(f'Loading ControlNet: {controlnet_model} ...')
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    )

    print(f'Loading SD inpainting base: {sd_model} ...')
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        sd_model,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,
    ).to(device)

    pipe.enable_attention_slicing()

    # LoRA fine-tune weights (optional — set lora_weights in config to enable)
    lora_weights = cfg.get('lora_weights', None)
    lora_scale   = float(cfg.get('lora_scale', 0.7))
    if lora_weights:
        print(f'Loading LoRA weights: {lora_weights} (scale={lora_scale}) ...')
        pipe.load_lora_weights(lora_weights, adapter_name="fundus")
pipe.set_adapters(["fundus"], adapter_weights=[lora_scale])
        print('✅ LoRA fused')

    _sd_pipe = {
        'pipe':   pipe,
        'device': device,
        'cfg':    cfg,
    }

    print(f'✅ SD + ControlNet loaded (device={device})')
    return _sd_pipe


def inpaint(image_rgb: np.ndarray,
            mask: np.ndarray,
            vessel_mask: np.ndarray = None,
            control_image: Image.Image = None,
            device: str = 'cuda',
            seed: int = None,
            controlnet_conditioning_scale: float = None,
            fov: tuple = None) -> np.ndarray:
    """
    Run SD + ControlNet inpainting on a single image.

    Args:
        image_rgb:                    uint8 (H, W, 3) RGB original fundus image
        mask:                         uint8 (H, W) inpaint mask (255 = fill)
        vessel_mask:                  uint8 (H, W) binary vessel mask — used to
                                      build the ControlNet conditioning image if
                                      control_image is not provided directly
        control_image:                PIL RGB — prebuilt ControlNet conditioning
                                      image. If None, built from vessel_mask.
        device:                       cuda | cpu
        seed:                         random seed. None = random per image.
        controlnet_conditioning_scale: override config value (used for sweep).
        fov:                          (cx, cy, r) tuple from preprocessing.
                                      If provided, SD output is hard-masked to
                                      inside the FOV — prevents teal cast and
                                      hallucinations in the dark border region.

    Returns:
        uint8 (H, W, 3) RGB de-identified image
    """
    if _sd_pipe is None:
        raise RuntimeError('SD pipeline not loaded. Call inpainting.load_model() first.')

    import torch
    from .vessel_pattern import build_control_image

    pipe = _sd_pipe['pipe']
    cfg  = _sd_pipe['cfg']

    prompt          = cfg.get('prompt', '').strip()
    negative_prompt = cfg.get('negative_prompt', '').strip()
    steps           = int(cfg.get('num_inference_steps', 50))
    guidance_scale  = float(cfg.get('guidance_scale', 7.5))
    strength        = float(cfg.get('strength', 1.0))
    cn_scale        = controlnet_conditioning_scale or float(
        cfg.get('controlnet_conditioning_scale', 0.6)
    )
    vp_cfg          = cfg.get('vessel_pattern', {})

    h, w = image_rgb.shape[:2]
    pil_image = Image.fromarray(image_rgb).resize((512, 512))
    pil_mask  = Image.fromarray(mask).resize((512, 512), resample=Image.NEAREST)

    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())

    # Build ControlNet conditioning image if not provided
    if control_image is None:
        if vessel_mask is None:
            raise ValueError(
                'Either vessel_mask or control_image must be provided for ControlNet inpainting.'
            )
        control_image = build_control_image(
            vessel_mask=vessel_mask,
            inpaint_mask=mask,
            cfg=vp_cfg,
            seed=seed,
        )
    # Resize conditioning image to 512×512 to match pipeline
    control_image = control_image.resize((512, 512), resample=Image.NEAREST)

    generator = torch.Generator(device=device).manual_seed(seed)

    result_pil = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        mask_image=pil_mask,
        control_image=control_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        controlnet_conditioning_scale=cn_scale,
        generator=generator,
    ).images[0]

    result_np = np.array(result_pil.resize((w, h), resample=Image.LANCZOS))

    # Hard-mask result to FOV — zero out anything SD generated outside the retina
    if fov is not None:
        cx, cy, r = fov
        fov_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(fov_mask, (cx, cy), r, 255, -1)
        fov_3ch = fov_mask[:, :, np.newaxis] / 255.0
        result_np = (result_np * fov_3ch).astype(np.uint8)

    return result_np.astype(np.uint8)
