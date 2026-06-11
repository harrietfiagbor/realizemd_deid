"""
pipeline/adversarial.py
Option C — Adversarial Privacy Pass.

Post-processing step applied after SD + ControlNet + LoRA inpainting.
Iterative gradient-based perturbation that pushes the de-identified image
away from the patient's original embedding(s) in RETFound + FLAIR space.

Design:
  - Per-image optimisation at inference time. No training run required.
  - Attacks multiple encoders simultaneously (RETFound + FLAIR) to improve
    transferability against non-RETFound attackers.
  - Perturbation clipped to L-inf epsilon budget to stay within perceptual limits.
  - Clinical signal (vessels, lesions) preserved via soft mask — perturbation
    is suppressed in lesion regions.

Threat model:
  Robust against RETFound-class and FLAIR-class attackers.
  Not guaranteed to transfer to encoders outside these two families.
  Frame as demo/paper privacy, not production guarantee (production = Path B).

Usage (standalone):
    from pipeline.adversarial import AdversarialPrivacyPass, load_encoders

    encoders = load_encoders(cfg, device)
    privacy_pass = AdversarialPrivacyPass(encoders, cfg)

    # anchor_rgb = original image before inpainting (patient's identity reference)
    # inpainted_rgb = output of SD + ControlNet + LoRA
    # lesion_mask = from pathology.detect_all (suppresses perturbation on lesions)
    perturbed = privacy_pass.run(
        inpainted_rgb=inpainted_rgb,
        anchor_rgb=anchor_rgb,
        lesion_mask=lesion_mask,
    )

Config (default.yaml under `adversarial:`):
    adversarial:
      enabled: true
      epsilon: 0.05          # L-inf budget as fraction of [0,1] range (~12/255)
      step_size: 0.005       # gradient step size per iteration
      n_steps: 100           # number of optimisation steps
      encoders:              # which encoders to attack
        retfound:
          enabled: true
          weights: models/RETFound_mae_natureCFP.pth
          dir:     /workspace/RETFound
        flair:
          enabled: true
          weights: models/flair/flair_pretrained.pth
          dir:     /workspace/FLAIR
      loss_weights:
        retfound: 1.0
        flair:    1.0
      lesion_suppression: true   # suppress perturbation on detected lesions
      lesion_suppress_weight: 0.1  # residual perturbation allowed on lesions
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Encoder loading
# ─────────────────────────────────────────────────────────────────────────────

_RETFOUND_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

_FLAIR_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _load_retfound(weights_path: str, retfound_dir: str, device: str):
    import sys
    if retfound_dir not in sys.path:
        sys.path.insert(0, retfound_dir)
    import models_vit

    constructor = None
    for name in ['RETFound_mae', 'RETFound_dinov2', 'vit_large_patch16',
                 'vit_large', 'create_model']:
        if hasattr(models_vit, name):
            constructor = getattr(models_vit, name)
            break
    if constructor is None:
        raise RuntimeError(
            f'Cannot find ViT constructor in models_vit. '
            f'Available: {[x for x in dir(models_vit) if not x.startswith("_")]}'
        )

    model = constructor(num_classes=0, global_pool=True)

    # timm compatibility patch
    try:
        import inspect
        orig_ff = model.forward_features
        sig = inspect.signature(orig_ff)
        if 'attn_mask' not in sig.parameters and 'kwargs' not in str(sig):
            def patched_ff(x, *args, **kwargs):
                return orig_ff(x)
            model.forward_features = patched_ff
    except Exception:
        pass

    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print(f'  ✅ RETFound encoder loaded (device={device})')
    return model


def _load_flair(weights_path: str, flair_dir: str, device: str):
    """
    Load FLAIR vision encoder via from_pretrained (weights auto-cached by HF).
    weights_path and flair_dir are ignored — kept for API consistency.
    """
    try:
        from flair import FLAIRModel
        model = FLAIRModel.from_pretrained('jusiro2/FLAIR')
        encoder = model.vision_encoder
        encoder.eval().to(device)
        print(f'  ✅ FLAIR encoder loaded (device={device})')
        return encoder, 'flair_native'
    except Exception as e:
        raise RuntimeError(
            f'Failed to load FLAIR: {e}. '
            f'Run: pip install git+https://github.com/jusiro/FLAIR.git'
        )


def load_encoders(cfg: dict, device: str) -> dict:
    """
    Load all configured encoders. Returns {name: {'model': ..., 'transform': ..., 'weight': ...}}
    """
    adv_cfg = cfg.get('adversarial', {})
    enc_cfg  = adv_cfg.get('encoders', {})
    wts      = adv_cfg.get('loss_weights', {})
    encoders = {}

    # RETFound
    rf_cfg = enc_cfg.get('retfound', {})
    if rf_cfg.get('enabled', True):
        rf_weights = rf_cfg.get('weights', 'models/RETFound_mae_natureCFP.pth')
        rf_dir     = rf_cfg.get('dir', '/workspace/RETFound')
        try:
            model = _load_retfound(rf_weights, rf_dir, device)
            encoders['retfound'] = {
                'model':     model,
                'transform': _RETFOUND_TRANSFORM,
                'weight':    float(wts.get('retfound', 1.0)),
            }
        except Exception as e:
            print(f'  ⚠️  RETFound load failed: {e}')

    # FLAIR
    fl_cfg = enc_cfg.get('flair', {})
    if fl_cfg.get('enabled', False):
        fl_weights = fl_cfg.get('weights', 'models/flair/flair_pretrained.pth')
        fl_dir     = fl_cfg.get('dir', '/workspace/FLAIR')
        try:
            model, _ = _load_flair(fl_weights, fl_dir, device)
            encoders['flair'] = {
                'model':     model,
                'transform': _FLAIR_TRANSFORM,
                'weight':    float(wts.get('flair', 1.0)),
            }
        except Exception as e:
            print(f'  ⚠️  FLAIR load failed (continuing with RETFound only): {e}')

    if not encoders:
        raise RuntimeError(
            'No encoders loaded. Check adversarial.encoders config and '
            'verify weights paths exist.'
        )

    print(f'  Encoders active: {list(encoders.keys())}')
    return encoders


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers (with gradient)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_tensor(model, transform, img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Embed a (1, 3, H, W) float tensor in [0, 1].
    Applies the encoder's normalisation transform and returns the embedding.
    Keeps gradients attached.
    """
    # Apply normalisation (mean/std) manually so gradients flow through img_tensor
    mean = torch.tensor(transform.transforms[-1].mean,
                        device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor(transform.transforms[-1].std,
                        device=img_tensor.device).view(1, 3, 1, 1)

    # Resize to 224×224
    resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear',
                            align_corners=False)
    normalised = (resized - mean) / std

    # Forward pass — keep grad
    if hasattr(model, 'forward_features'):
        emb = model.forward_features(normalised)
    else:
        emb = model(normalised)

    emb = emb.squeeze(0)
    if emb.ndim > 1:
        emb = emb[0]  # CLS token

    return emb  # (D,)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Privacy Pass
# ─────────────────────────────────────────────────────────────────────────────

class AdversarialPrivacyPass:
    """
    Per-image adversarial perturbation against one or more retinal encoders.

    Loss: maximise cosine distance between perturbed image embedding
          and anchor (original patient) embedding, across all encoders.

    L-inf constraint: perturbation clipped to [-epsilon, +epsilon] each step.
    Lesion suppression: perturbation masked on detected lesion regions so
    clinical signal is preserved.
    """

    def __init__(self, encoders: dict, cfg: dict):
        adv_cfg = cfg.get('adversarial', {})
        self.encoders   = encoders
        self.epsilon    = float(adv_cfg.get('epsilon',    0.05))
        self.step_size  = float(adv_cfg.get('step_size',  0.005))
        self.n_steps    = int(adv_cfg.get('n_steps',      100))
        self.lesion_sup = adv_cfg.get('lesion_suppression', True)
        self.lesion_w   = float(adv_cfg.get('lesion_suppress_weight', 0.1))

    def _to_tensor(self, img_rgb: np.ndarray, device: str) -> torch.Tensor:
        """uint8 (H, W, 3) → float (1, 3, H, W) in [0, 1]."""
        t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """float (1, 3, H, W) in [0, 1] → uint8 (H, W, 3)."""
        arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def _get_anchor_embeddings(self, anchor_rgb: np.ndarray,
                                device: str) -> dict:
        """Embed the original (anchor) image with all encoders. No grad needed."""
        anchor_tensor = self._to_tensor(anchor_rgb, device)
        anchors = {}
        for name, enc in self.encoders.items():
            anchors[name] = _embed_tensor(
                enc['model'], enc['transform'], anchor_tensor
            ).detach()
        return anchors

    def _build_perturbation_mask(self, lesion_mask: Optional[np.ndarray],
                                  h: int, w: int, device: str) -> torch.Tensor:
        """
        Build a (1, 1, H, W) float mask controlling where perturbation is allowed.
        1.0 = full perturbation, lesion_suppress_weight = suppressed on lesions.
        """
        mask = torch.ones(1, 1, h, w, device=device)
        if self.lesion_sup and lesion_mask is not None:
            lm = torch.from_numpy(
                (lesion_mask > 0).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)
            # Resize to match image if needed
            if lm.shape[2:] != (h, w):
                lm = F.interpolate(lm, size=(h, w), mode='nearest')
            mask = mask * (1.0 - lm) + lm * self.lesion_w
        return mask

    def run(self,
            inpainted_rgb: np.ndarray,
            anchor_rgb: np.ndarray,
            lesion_mask: Optional[np.ndarray] = None,
            device: str = 'cuda') -> np.ndarray:
        """
        Apply adversarial perturbation to the inpainted image.

        Args:
            inpainted_rgb:  uint8 (H, W, 3) — output of SD + ControlNet + LoRA
            anchor_rgb:     uint8 (H, W, 3) — original image (identity reference)
            lesion_mask:    uint8 (H, W) — combined lesion mask from pathology module.
                            Perturbation suppressed here to preserve clinical signal.
            device:         cuda | cpu

        Returns:
            uint8 (H, W, 3) perturbed image
        """
        h, w = inpainted_rgb.shape[:2]

        # Set all encoder models to eval, no grad on weights
        for enc in self.encoders.values():
            enc['model'].eval()
            for p in enc['model'].parameters():
                p.requires_grad_(False)

        # Anchor embeddings (no grad)
        anchor_embs = self._get_anchor_embeddings(anchor_rgb, device)

        # Perturbable image tensor
        x0 = self._to_tensor(inpainted_rgb, device)
        delta = torch.zeros_like(x0, requires_grad=False)

        # Perturbation mask
        ptb_mask = self._build_perturbation_mask(lesion_mask, h, w, device)

        for step in range(self.n_steps):
            delta.requires_grad_(True)

            x_ptb = torch.clamp(x0 + delta * ptb_mask, 0.0, 1.0)

            # Loss: sum of cosine similarities across encoders (we want to minimise)
            # Minimising similarity = maximising distance from anchor
            loss = torch.tensor(0.0, device=device)
            for name, enc in self.encoders.items():
                emb_ptb = _embed_tensor(enc['model'], enc['transform'], x_ptb)
                sim = _cosine_sim(emb_ptb, anchor_embs[name])
                loss = loss + enc['weight'] * sim

            loss.backward()

            with torch.no_grad():
                # Gradient descent step (minimise similarity = move away from anchor)
                grad = delta.grad.detach()
                grad_sign = grad.sign()

                delta_new = delta.detach() - self.step_size * grad_sign * ptb_mask

                # L-inf clip
                delta_new = torch.clamp(delta_new, -self.epsilon, self.epsilon)

                # Ensure x0 + delta stays in [0, 1]
                delta_new = torch.clamp(x0 + delta_new, 0.0, 1.0) - x0

                delta = delta_new

            # Free graph
            if delta.grad is not None:
                delta.grad.zero_()

        x_final = torch.clamp(x0 + delta * ptb_mask, 0.0, 1.0)
        return self._to_numpy(x_final)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run pass on a single image given loaded state
# ─────────────────────────────────────────────────────────────────────────────

_privacy_pass = None


def load_privacy_pass(cfg: dict, device: str) -> AdversarialPrivacyPass:
    """Load encoders and initialise the privacy pass. Call once at startup."""
    global _privacy_pass
    print('Loading adversarial privacy pass encoders ...')
    encoders = load_encoders(cfg, device)
    _privacy_pass = AdversarialPrivacyPass(encoders, cfg)
    print('✅ Adversarial privacy pass ready')
    return _privacy_pass


def apply_privacy_pass(inpainted_rgb: np.ndarray,
                       anchor_rgb: np.ndarray,
                       lesion_mask: Optional[np.ndarray] = None,
                       device: str = 'cuda') -> np.ndarray:
    """Apply the loaded privacy pass. Call load_privacy_pass() first."""
    if _privacy_pass is None:
        raise RuntimeError(
            'Privacy pass not loaded. Call adversarial.load_privacy_pass() first.'
        )
    return _privacy_pass.run(
        inpainted_rgb=inpainted_rgb,
        anchor_rgb=anchor_rgb,
        lesion_mask=lesion_mask,
        device=device,
    )
