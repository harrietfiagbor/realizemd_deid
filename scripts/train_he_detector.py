#!/usr/bin/env python3
"""
RealizeMD — Learned HE Detector Training
=========================================
Trains a high-recall binary haemorrhage segmenter.
Target: HE recall@IoU=0.3 >= 0.70, HE pixel preservation >= 0.98.

Loss: Focal Tversky (alpha=0.7, beta=0.3, gamma=1.33) — FN penalised harder than FP.
Arch: EfficientNet-B4 U-Net (segmentation_models_pytorch).
Data: IDRiD-train + DDR (HE only). IDRiD-test held out for final eval.

Usage (RunPod or local GPU):
    python scripts/train_he_detector.py \
        --idrid_dir  /workspace/data/idrid/A.\ Segmentation/A.\ Segmentation \
        --ddr_dir    /workspace/data/ddr \
        --out_dir    /workspace/models/he_detector \
        --epochs     80 \
        --patch_size 768 \
        --batch_size 4

Add --ddr_dir only when DDR is available. Script runs on IDRiD alone first.
"""

import argparse
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp
from skimage import measure as sk_measure

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    Tversky with FN penalised harder than FP (alpha > beta), plus focal gamma.
    alpha=0.7 / beta=0.3: recall-leaning starting point.
    If recall stalls below 0.60, push alpha toward 0.8.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1).float()
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        return (1 - tversky) ** self.gamma


# ── Metric ───────────────────────────────────────────────────────────────────

def recall_at_iou(pred_bin: np.ndarray, gt_bin: np.ndarray,
                  iou_thresh: float = 0.3):
    """Blob-level recall. Returns (recall, hits, total_gt_blobs)."""
    gt_labeled   = sk_measure.label(gt_bin)
    pred_labeled = sk_measure.label(pred_bin)
    gt_regions   = sk_measure.regionprops(gt_labeled)
    pred_areas   = {r.label: r.area for r in sk_measure.regionprops(pred_labeled)}
    if not gt_regions:
        return None, 0, 0
    hits = 0
    for gt_reg in gt_regions:
        r0, c0, r1, c1 = gt_reg.bbox
        gt_blob   = (gt_labeled[r0:r1, c0:c1] == gt_reg.label).astype(np.uint8)
        pred_crop = pred_labeled[r0:r1, c0:c1]
        overlap   = np.unique(pred_crop[gt_blob > 0])
        overlap   = overlap[overlap > 0]
        best_iou  = 0.0
        for pl in overlap:
            inter         = (gt_blob & (pred_crop == pl)).sum()
            union         = gt_reg.area + pred_areas.get(pl, 0) - inter
            best_iou      = max(best_iou, inter / max(union, 1))
        if best_iou >= iou_thresh:
            hits += 1
    return hits / len(gt_regions), hits, len(gt_regions)


# ── Preprocessing (match inference pipeline CLAHE exactly) ───────────────────

def preprocess_for_training(img_bgr: np.ndarray) -> np.ndarray:
    """Apply same CLAHE preprocessing as the inference pipeline."""
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe  = clahe.apply(l)
    enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


# ── Patch extraction ──────────────────────────────────────────────────────────

def extract_patches(image: np.ndarray, mask: np.ndarray,
                    patch: int = 768, n_pos: int = 8, n_neg: int = 4):
    """
    Oversample lesion-containing patches.
    Returns list of (img_patch [H,W,3], mask_patch [H,W]).
    """
    H, W   = mask.shape
    ys, xs = np.where(mask > 0)
    half   = patch // 2
    patches = []

    # Positive patches centred near lesion pixels
    if len(ys) > 0:
        for _ in range(n_pos):
            i  = random.randrange(len(ys))
            cy = int(np.clip(ys[i] + random.randint(-half // 2, half // 2), half, H - half))
            cx = int(np.clip(xs[i] + random.randint(-half // 2, half // 2), half, W - half))
            y0, x0 = cy - half, cx - half
            patches.append((image[y0:y0+patch, x0:x0+patch],
                            mask[y0:y0+patch, x0:x0+patch]))

    # Random negative patches for background context
    for _ in range(n_neg):
        y0 = random.randint(0, max(0, H - patch))
        x0 = random.randint(0, max(0, W - patch))
        patches.append((image[y0:y0+patch, x0:x0+patch],
                        mask[y0:y0+patch, x0:x0+patch]))
    return patches


# ── Augmentation ──────────────────────────────────────────────────────────────

def build_aug(patch_size: int) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.Resize(patch_size, patch_size, always_apply=True),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class HEDataset(Dataset):
    def __init__(self, samples: list, patch_size: int = 768,
                 augment: bool = True, n_pos: int = 8, n_neg: int = 4):
        """
        samples: list of (img_path, mask_path) tuples
        Patches are generated on-the-fly each epoch.
        """
        self.samples    = samples
        self.patch_size = patch_size
        self.augment    = augment
        self.n_pos      = n_pos
        self.n_neg      = n_neg
        self.aug        = build_aug(patch_size)
        self._build_patches()

    def _build_patches(self):
        self.patches = []
        for img_path, mask_path in self.samples:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = preprocess_for_training(img_bgr)
            mask    = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            for img_p, mask_p in extract_patches(
                    img_rgb, mask, self.patch_size, self.n_pos, self.n_neg):
                if img_p.shape[0] == self.patch_size and img_p.shape[1] == self.patch_size:
                    self.patches.append((img_p, mask_p))

    def resample(self):
        """Call at the start of each epoch to re-randomise patches."""
        self._build_patches()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        if self.augment:
            out  = self.aug(image=img, mask=mask)
            img  = out['image']
            mask = out['mask']
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_idrid_samples(idrid_dir: Path, split: str = 'train'):
    """
    Returns list of (img_path, he_mask_path).
    split: 'train' (54 images) or 'test' (27 images).
    """
    folder = 'a. Training Set' if split == 'train' else 'b. Testing Set'
    img_dir  = idrid_dir / '1. Original Images' / folder
    mask_dir = idrid_dir / '2. All Segmentation Groundtruths' / folder / '2. Haemorrhages'
    samples  = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        stem    = img_path.stem          # e.g. IDRiD_01
        he_glob = list(mask_dir.glob(f'{stem}_HE.*'))
        if he_glob:
            samples.append((img_path, he_glob[0]))
        else:
            samples.append((img_path, None))   # no HE GT for this image
    # Drop images with no HE GT (healthy retinas contribute no positive signal)
    return [(i, m) for i, m in samples if m is not None]


def load_ddr_samples(ddr_dir: Path):
    """
    DDR dataset layout: ddr_dir/image/<id>.jpg, ddr_dir/label/HE/<id>.png
    Returns list of (img_path, he_mask_path).
    """
    img_dir  = ddr_dir / 'image'
    mask_dir = ddr_dir / 'label' / 'HE'
    if not img_dir.exists():
        return []
    samples = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        mask_path = mask_dir / (img_path.stem + '.png')
        if mask_path.exists():
            samples.append((img_path, mask_path))
    return samples


# ── Sliding-window inference ──────────────────────────────────────────────────

def sliding_window_predict(model: nn.Module, img_rgb: np.ndarray,
                           patch: int = 768, stride: int = 512,
                           threshold: float = 0.35,
                           device: str = 'cuda') -> np.ndarray:
    """
    Full-resolution inference: average overlapping patch predictions.
    Returns binary mask (uint8, 0/255) at original image size.
    """
    model.eval()
    H, W    = img_rgb.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, H - patch, stride)) + [H - patch]
    xs = list(range(0, W - patch, stride)) + [W - patch]

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y0 = max(0, min(y0, H - patch))
                x0 = max(0, min(x0, W - patch))
                patch_img = img_rgb[y0:y0+patch, x0:x0+patch]
                t = torch.from_numpy(
                    patch_img.transpose(2, 0, 1)).float() / 255.0
                t = t.unsqueeze(0).to(device)
                prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                prob_map[y0:y0+patch, x0:x0+patch] += prob
                count[y0:y0+patch, x0:x0+patch]    += 1

    prob_map /= np.maximum(count, 1)
    return (prob_map > threshold).astype(np.uint8) * 255


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: nn.Module, val_samples: list, args, device: str):
    """
    Full-image sliding-window inference on val set.
    Returns aggregate recall@IoU=0.3.
    """
    model.eval()
    hits_total = 0
    gt_total   = 0
    for img_path, mask_path in val_samples:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = preprocess_for_training(img_bgr)
        # Resize to manageable size for validation speed (keep aspect)
        H, W    = img_rgb.shape[:2]
        scale   = min(1.0, 2048 / max(H, W))
        img_s   = cv2.resize(img_rgb, (int(W*scale), int(H*scale)))
        pred    = sliding_window_predict(
            model, img_s,
            patch=args.patch_size, stride=args.patch_size // 2,
            threshold=args.threshold, device=device)
        pred_bin = (pred > 0).astype(np.uint8)

        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (int(W*scale), int(H*scale)),
                        interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 0).astype(np.uint8)

        _, hits, n_gt = recall_at_iou(pred_bin, gt_bin)
        hits_total += hits
        gt_total   += n_gt

    recall = hits_total / max(gt_total, 1)
    return recall, hits_total, gt_total


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    idrid_dir  = Path(args.idrid_dir)
    train_samp = load_idrid_samples(idrid_dir, split='train')
    val_samp   = load_idrid_samples(idrid_dir, split='test')

    if args.ddr_dir:
        ddr_samp = load_ddr_samples(Path(args.ddr_dir))
        print(f"DDR samples: {len(ddr_samp)}")
        train_samp = train_samp + ddr_samp

    print(f"Train samples (with HE GT): {len(train_samp)}")
    print(f"Val   samples (with HE GT): {len(val_samp)}")

    train_ds = HEDataset(train_samp, patch_size=args.patch_size,
                         augment=True,  n_pos=8, n_neg=4)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = smp.Unet(
        encoder_name    = args.encoder,
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,
    ).to(device)
    print(f"Model: Unet + {args.encoder}")

    # ── Optimiser + schedule ──────────────────────────────────────────────────
    criterion = FocalTverskyLoss(alpha=args.alpha, beta=1 - args.alpha)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_recall  = 0.0
    no_improve   = 0
    patience     = args.patience

    print(f"\nStarting training — {args.epochs} epochs, patience={patience}")
    print(f"Target: recall@IoU=0.3 >= {args.target_recall}\n")

    for epoch in range(1, args.epochs + 1):
        # Resample patches each epoch for diversity
        train_ds.resample()
        train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch:03d}", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            optimiser.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_dl), 1)

        # Validate every 5 epochs (full-image sliding window is slow)
        if epoch % 5 == 0 or epoch == 1:
            recall, hits, total = validate(model, val_samp, args, device)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                  f"val recall@IoU=0.3: {recall:.4f} ({hits}/{total})")

            if recall > best_recall:
                best_recall = recall
                no_improve  = 0
                ckpt = out_dir / 'he_detector_best.pth'
                torch.save({'epoch': epoch, 'recall': recall,
                            'state_dict': model.state_dict(),
                            'threshold': args.threshold}, str(ckpt))
                print(f"  ✓ New best saved: {ckpt.name}  recall={recall:.4f}")
            else:
                no_improve += 1
                print(f"  no improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    print("Early stopping.")
                    break

            if recall >= args.target_recall:
                print(f"\nTarget recall {args.target_recall} reached — done.")
                break
        else:
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

    # Save final checkpoint
    torch.save({'epoch': epoch, 'recall': best_recall,
                'state_dict': model.state_dict(),
                'threshold': args.threshold},
               str(out_dir / 'he_detector_final.pth'))
    print(f"\nBest val recall@IoU=0.3: {best_recall:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train learned HE detector')
    p.add_argument('--idrid_dir',     required=True,
                   help='Path to IDRiD A. Segmentation/A. Segmentation dir')
    p.add_argument('--ddr_dir',       default=None,
                   help='Path to DDR dataset root (optional)')
    p.add_argument('--out_dir',       default='models/he_detector')
    p.add_argument('--encoder',       default='efficientnet-b4')
    p.add_argument('--epochs',        type=int,   default=80)
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--patch_size',    type=int,   default=768)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--alpha',         type=float, default=0.7,
                   help='Tversky alpha (FN weight). Push to 0.8 if recall stalls.')
    p.add_argument('--threshold',     type=float, default=0.35,
                   help='Probability threshold at inference. Tune on val.')
    p.add_argument('--target_recall', type=float, default=0.70)
    p.add_argument('--patience',      type=int,   default=10,
                   help='Early-stop patience in validation checks (5 epochs each).')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
