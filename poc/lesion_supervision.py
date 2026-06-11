"""
Disease-subspace supervision module.

Replaces the binary `disease_head` + `dis_bce` in poc_model.py with:
  - A small mask decoder: z_dis -> predicted HE / EX / MA masks at 256x256
  - A DR grade head:      z_dis -> retinopathy grade (0-4)
  - Loss: soft Dice per mask channel + cross-entropy for DR grade

Why Dice loss for masks:
  HE/EX/MA lesions are sparse — they occupy a tiny fraction of pixels.
  Binary cross-entropy would be dominated by the background (>99% of pixels).
  Dice loss is invariant to class imbalance and directly optimises overlap,
  which matches what we want: z_dis must capture lesion location and shape.

Interface with poc_model.py:
  1. Replace  self.disease_head = nn.Linear(dis_dim, 1)
     with      self.disease_head = LesionSupervisionHead(dis_dim, mask_size=256)

  2. In forward(), replace
       dlogit = vae.disease_head(z_dis)
     with
       mask_logits, grade_logit = vae.disease_head(z_dis)

  3. Replace vae_loss(...) call with lesion_supervision_loss(...) which
     accepts the real masks and grade labels instead of a binary label.

  The rest of the VAE (encoder, decoder, dCor, KL) stays identical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Mask decoder ──────────────────────────────────────────────────────────────

class LesionSupervisionHead(nn.Module):
    """
    Decodes z_dis (flat vector, dim=dis_dim) into:
      - mask_logits : (B, 3, mask_size, mask_size)  — HE / EX / MA channels
      - grade_logit : (B, 5)                        — DR grade 0-4

    mask_size should match the resolution masks are resized to before passing
    to the loss. Default 256 is a practical balance: captures lesion locations
    without requiring a large decoder.

    Args:
        dis_dim   : dimension of z_dis (8 in the toy; larger in the real model)
        mask_size : spatial resolution of predicted and target masks (default 256)
    """
    def __init__(self, dis_dim: int, mask_size: int = 256):
        super().__init__()
        self.mask_size = mask_size

        # Project z_dis to a small spatial feature map
        self.fc = nn.Linear(dis_dim, 128 * 8 * 8)

        # Upsample 8x8 -> mask_size via transposed convolutions
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256  (5 doublings = 2^5 * 8 = 256)
        self.mask_dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(64,  32, 4, 2, 1), nn.ReLU(),   # 32
            nn.ConvTranspose2d(32,  16, 4, 2, 1), nn.ReLU(),   # 64
            nn.ConvTranspose2d(16,   8, 4, 2, 1), nn.ReLU(),   # 128
            nn.ConvTranspose2d( 8,   3, 4, 2, 1),              # 256, 3 channels (HE/EX/MA)
        )

        # DR grade classifier: 5 classes (grade 0-4)
        self.grade_head = nn.Linear(dis_dim, 5)

    def forward(self, z_dis: torch.Tensor):
        """
        Args:
            z_dis : (B, dis_dim)
        Returns:
            mask_logits : (B, 3, mask_size, mask_size)  — raw logits, apply sigmoid
            grade_logit : (B, 5)                        — raw logits, apply softmax
        """
        h = F.relu(self.fc(z_dis)).view(-1, 128, 8, 8)
        mask_logits = self.mask_dec(h)
        grade_logit = self.grade_head(z_dis)
        return mask_logits, grade_logit


# ── Dice loss ─────────────────────────────────────────────────────────────────

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor,
                   smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss averaged over batch and channels.

    Args:
        logits  : (B, C, H, W) raw logits
        targets : (B, C, H, W) binary float masks in [0, 1]
    """
    probs = torch.sigmoid(logits)
    # flatten spatial dims
    p = probs.flatten(2)   # (B, C, H*W)
    t = targets.flatten(2) # (B, C, H*W)
    inter = (p * t).sum(2)
    union = p.sum(2) + t.sum(2)
    dice  = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


# ── Combined supervision loss ─────────────────────────────────────────────────

def lesion_supervision_loss(mask_logits: torch.Tensor,
                            grade_logit: torch.Tensor,
                            he_mask: torch.Tensor,
                            ex_mask: torch.Tensor,
                            ma_mask: torch.Tensor,
                            dr_grade: torch.Tensor,
                            lam_grade: float = 0.5) -> tuple:
    """
    Combined mask Dice loss + DR grade cross-entropy.

    Args:
        mask_logits : (B, 3, H, W)  — predicted HE/EX/MA logits
        grade_logit : (B, 5)        — predicted DR grade logits
        he_mask     : (B, H, W)     — ground-truth HE  binary mask
        ex_mask     : (B, H, W)     — ground-truth EX  binary mask
        ma_mask     : (B, H, W)     — ground-truth MA  binary mask
        dr_grade    : (B,)  long    — ground-truth DR grade 0-4
        lam_grade   : weight for grade loss relative to Dice loss

    Returns:
        total loss (scalar), dict of components
    """
    # Stack masks to (B, 3, H, W) to match mask_logits channels
    targets = torch.stack([he_mask, ex_mask, ma_mask], dim=1)  # (B, 3, H, W)

    # Resize targets to match mask_logits spatial size if needed
    H, W = mask_logits.shape[2], mask_logits.shape[3]
    if targets.shape[2] != H or targets.shape[3] != W:
        targets = F.interpolate(targets, size=(H, W), mode="nearest")

    dice = soft_dice_loss(mask_logits, targets)

    # Only compute grade loss for samples that have a valid grade (>= 0)
    valid = (dr_grade >= 0)
    if valid.any():
        grade_loss = F.cross_entropy(grade_logit[valid], dr_grade[valid])
    else:
        grade_loss = torch.tensor(0.0, device=mask_logits.device)

    total = dice + lam_grade * grade_loss
    return total, dict(dice=dice.item(), grade=grade_loss.item())


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulate a batch: B=4, dis_dim=32 (larger than toy's 8 for real use)
    B, dis_dim = 4, 32
    z_dis = torch.randn(B, dis_dim)

    head = LesionSupervisionHead(dis_dim=dis_dim, mask_size=256)
    mask_logits, grade_logit = head(z_dis)
    print(f"mask_logits : {tuple(mask_logits.shape)}")   # (4, 3, 256, 256)
    print(f"grade_logit : {tuple(grade_logit.shape)}")   # (4, 5)

    # Simulate ground-truth masks (sparse — most pixels zero)
    he = (torch.rand(B, 256, 256) > 0.98).float()
    ex = (torch.rand(B, 256, 256) > 0.95).float()
    ma = (torch.rand(B, 256, 256) > 0.99).float()
    grades = torch.tensor([3, 2, 3, 4])

    loss, parts = lesion_supervision_loss(mask_logits, grade_logit,
                                          he, ex, ma, grades)
    print(f"loss={loss.item():.4f}  dice={parts['dice']:.4f}  grade={parts['grade']:.4f}")
    print("OK")
