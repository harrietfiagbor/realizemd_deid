#!/usr/bin/env python3
"""
RealizeMD — Learned Optic Disc (OD) Detector Training
========================================================
Trains a binary optic-disc segmenter to replace the brightness-threshold
heuristic in `detect_optic_disc()` (pathology.py), which fails ~33% of the
time on EyePACS (wrong-blob selection on glare/haze/bright-lesion cases —
see decision.md "Optic Disc Detector: EyePACS Wrong-Blob Failure").

Unlike EX/HE/MA, the disc is one large, high-contrast, consistently-shaped
blob per image — not many small sparse lesions. So this script trains on
whole-image resizes rather than sliding-window patches, and does not assume
efficientnet-b2 is the right encoder just because it was for the lesion
detectors (that comparison was never run for this task — see decision.md
"Is b2 best for OD?"). --encoder is a required-thought CLI flag with no
implicit "obviously correct" default; try b0 (lighter, likely sufficient
given the easier task and larger combined dataset) against b2 explicitly.

Data: IDRiD (54 train + 27 test, OD GT) + REFUGE (400 train + 400 val,
Disc_Cup_Masks) + G1020 (1020 images, Masks/). Each dataset encodes the
disc mask differently — unified via `load_disc_mask()`:
  - IDRiD:  plain binary mask, disc = (pixel > 0)
  - REFUGE: 3-value mask (0=cup, 128=rim, 255=background), disc = (pixel < 255)
  - G1020:  3-value mask (0=background, 1/2=rim/cup), disc = (pixel > 0)
All three reduce to "disc = non-background pixels" — cup-vs-rim distinction
is discarded since only the outer disc boundary matters for exclusion.
IDRiD-test is held out as the primary validation set (matches every other
detector trained in this project). REFUGE-val and all of G1020 are folded
into training (G1020 has no established train/test split of its own).

Usage (RunPod or local GPU):
    python scripts/train_od_detector.py \
        --idrid_dir  "/workspace/data/idrid/A. Segmentation/A. Segmentation" \
        --refuge_dir /workspace/data/REFUGE \
        --g1020_dir  /workspace/data/G1020 \
        --out_dir    /workspace/models/od_detector \
        --encoder    efficientnet-b0 \
        --epochs     60
"""

import argparse
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


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """Tversky with FN penalised harder than FP, plus focal gamma."""
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 gamma: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        return (1 - tversky) ** self.gamma


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        inter   = (probs * targets).sum()
        return 1 - (2 * inter + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )


# ── Metrics ──────────────────────────────────────────────────────────────────

def dice_coef(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = int((pred_bin & gt_bin).sum())
    denom = int(pred_bin.sum()) + int(gt_bin.sum())
    return 1.0 if denom == 0 else (2.0 * inter) / denom


def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    return 1.0 if union == 0 else inter / union


# ── Preprocessing (match inference pipeline CLAHE exactly) ───────────────────

def preprocess_for_training(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe  = clahe.apply(l)
    enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


# ── Disc mask extraction (per-dataset encoding, unified to one rule) ─────────

def load_disc_mask(mask_path: Path, dataset: str) -> np.ndarray:
    """Returns binary (0/255) disc mask regardless of source encoding."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(mask_path)
    if dataset == 'refuge':
        disc = raw < 255          # 0=cup, 128=rim, 255=background
    else:                          # 'idrid' or 'g1020'
        disc = raw > 0            # background = 0
    return (disc.astype(np.uint8)) * 255


# ── Augmentation (whole-image, no patch extraction — disc is one big blob) ───
# No ElasticTransform: the disc's near-constant round/oval shape is a useful
# prior for this task, unlike sparse lesions — heavy warping would fight it.

def build_aug(img_size: int) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.Resize(img_size, img_size, always_apply=True),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ODDataset(Dataset):
    def __init__(self, samples: list, img_size: int = 512, augment: bool = True):
        """samples: list of (img_path, mask_path, dataset_tag) tuples."""
        self.samples  = samples
        self.img_size = img_size
        self.augment  = augment
        self.aug      = build_aug(img_size)
        self.resize_only = A.Compose([A.Resize(img_size, img_size, always_apply=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, dataset = self.samples[idx]
        img_bgr = cv2.imread(str(img_path))
        img_rgb = preprocess_for_training(img_bgr)
        mask    = load_disc_mask(mask_path, dataset)

        transform = self.aug if self.augment else self.resize_only
        out  = transform(image=img_rgb, mask=mask)
        img  = out['image']
        mask = out['mask']

        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_idrid_od_samples(idrid_dir: Path, split: str = 'train'):
    folder   = 'a. Training Set' if split == 'train' else 'b. Testing Set'
    img_dir  = idrid_dir / '1. Original Images' / folder
    mask_dir = idrid_dir / '2. All Segmentation Groundtruths' / folder / '5. Optic Disc'
    samples  = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        stem = img_path.stem
        od_glob = list(mask_dir.glob(f'{stem}_OD.*'))
        if od_glob:
            samples.append((img_path, od_glob[0], 'idrid'))
    return samples


def load_refuge_od_samples(refuge_dir: Path, split: str = 'train'):
    """
    split: 'train' -> Training400/{Glaucoma,Non-Glaucoma} + matching
           Annotation-Training400/Disc_Cup_Masks/{Glaucoma,Non-Glaucoma}
           'validation' -> REFUGE-Validation400 images + REFUGE-Validation400-GT masks
    """
    samples = []
    if split == 'train':
        img_root  = refuge_dir / 'Training400'
        mask_root = refuge_dir / 'Annotation-Training400' / 'Disc_Cup_Masks'
        for cls in ('Glaucoma', 'Non-Glaucoma'):
            img_dir  = img_root / cls
            mask_dir = mask_root / cls
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob('*.jpg')):
                mask_path = mask_dir / (img_path.stem + '.bmp')
                if mask_path.exists():
                    samples.append((img_path, mask_path, 'refuge'))
    elif split == 'validation':
        img_dir  = refuge_dir / 'REFUGE-Validation400'
        mask_dir = refuge_dir / 'REFUGE-Validation400-GT' / 'Disc_Cup_Masks'
        if img_dir.exists() and mask_dir.exists():
            for img_path in sorted(img_dir.glob('*.jpg')):
                mask_path = mask_dir / (img_path.stem + '.bmp')
                if mask_path.exists():
                    samples.append((img_path, mask_path, 'refuge'))
    return samples


def load_g1020_od_samples(g1020_dir: Path):
    img_dir  = g1020_dir / 'Images'
    mask_dir = g1020_dir / 'Masks'
    samples  = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        mask_path = mask_dir / (img_path.stem + '.png')
        if mask_path.exists():
            samples.append((img_path, mask_path, 'g1020'))
    return samples


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: nn.Module, val_samples: list, img_size: int, device: str):
    """Whole-image resize inference vs IDRiD-test GT. Returns (mean_dice, mean_iou)."""
    model.eval()
    dice_vals, iou_vals = [], []
    resize_only = A.Compose([A.Resize(img_size, img_size, always_apply=True)])
    with torch.no_grad():
        for img_path, mask_path, dataset in val_samples:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = preprocess_for_training(img_bgr)
            gt_mask = load_disc_mask(mask_path, dataset)

            out  = resize_only(image=img_rgb, mask=gt_mask)
            img  = out['image']
            gt_r = (out['mask'] > 0).astype(np.uint8)

            t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            prob = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
            pred_bin = (prob > 0.5).astype(np.uint8)

            dice_vals.append(dice_coef(pred_bin, gt_r))
            iou_vals.append(iou_score(pred_bin, gt_r))

    return float(np.mean(dice_vals)) if dice_vals else 0.0, \
           float(np.mean(iou_vals)) if iou_vals else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  Seed: {args.seed}  Encoder: {args.encoder}")

    # ── Data ──────────────────────────────────────────────────────────────────
    idrid_dir = Path(args.idrid_dir)
    train_samp = load_idrid_od_samples(idrid_dir, split='train')
    val_samp   = load_idrid_od_samples(idrid_dir, split='test')
    print(f"IDRiD  train={len(train_samp)}  test(val)={len(val_samp)}")

    if args.refuge_dir:
        refuge_dir = Path(args.refuge_dir)
        refuge_train = load_refuge_od_samples(refuge_dir, split='train')
        refuge_val   = load_refuge_od_samples(refuge_dir, split='validation')
        print(f"REFUGE train={len(refuge_train)}  validation={len(refuge_val)} (both -> training pool)")
        train_samp = train_samp + refuge_train + refuge_val

    if args.g1020_dir:
        g1020_samp = load_g1020_od_samples(Path(args.g1020_dir))
        print(f"G1020  samples={len(g1020_samp)} (-> training pool, no established split)")
        train_samp = train_samp + g1020_samp

    print(f"Total train samples: {len(train_samp)}")
    print(f"Val samples (IDRiD-test, held out): {len(val_samp)}")

    train_ds = ODDataset(train_samp, img_size=args.img_size, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = smp.Unet(
        encoder_name    = args.encoder,
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,
    ).to(device)
    print(f"Model: Unet + {args.encoder}")

    # ── Optimiser + schedule ──────────────────────────────────────────────────
    criterion_ft   = FocalTverskyLoss(alpha=args.alpha, beta=1 - args.alpha)
    criterion_dice = DiceLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_dice   = 0.0
    no_improve  = 0
    start_epoch = 1
    patience    = args.patience

    if args.resume:
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r['state_dict'])
        best_dice   = ckpt_r.get('dice', 0.0)
        start_epoch = ckpt_r.get('epoch', 0) + 1
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1}  best_dice={best_dice:.4f}")

    print(f"\nStarting training — {args.epochs} epochs, patience={patience}")
    print(f"Target: dice >= {args.target_dice}\n")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch:03d}", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            optimiser.zero_grad()
            logits = model(imgs)
            loss   = criterion_ft(logits, masks) + criterion_dice(logits, masks)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_dl), 1)

        if epoch % 5 == 0 or epoch == 1:
            mean_dice, mean_iou = validate(model, val_samp, args.img_size, device)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                  f"dice={mean_dice:.4f} | iou={mean_iou:.4f}")

            if mean_dice > best_dice:
                best_dice  = mean_dice
                no_improve = 0
                ckpt = out_dir / 'od_detector_best.pth'
                torch.save({'epoch': epoch, 'dice': mean_dice, 'iou': mean_iou,
                            'state_dict': model.state_dict(),
                            'encoder': args.encoder, 'img_size': args.img_size,
                            'seed': args.seed}, str(ckpt))
                print(f"  New best saved: dice={mean_dice:.4f}  iou={mean_iou:.4f}")
                if args.drive_dir:
                    import subprocess as _sp
                    r = _sp.run(['rclone', 'copy', str(ckpt), args.drive_dir],
                                capture_output=True, text=True)
                    if r.returncode == 0:
                        print(f"  Backed up to Drive: {args.drive_dir}")
                    else:
                        print(f"  Drive backup failed: {r.stderr.strip()}")
            else:
                no_improve += 1
                print(f"  no improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    print("Early stopping.")
                    break

            if mean_dice >= args.target_dice:
                print(f"\nTarget dice {args.target_dice} reached — done.")
                break
        else:
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

    torch.save({'epoch': epoch, 'dice': best_dice,
                'state_dict': model.state_dict(),
                'encoder': args.encoder, 'img_size': args.img_size},
               str(out_dir / 'od_detector_final.pth'))
    print(f"\nBest dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train learned optic disc detector')
    p.add_argument('--idrid_dir',  required=True,
                   help='Path to IDRiD A. Segmentation/A. Segmentation dir')
    p.add_argument('--refuge_dir', default=None,
                   help='Path to REFUGE dataset root (Training400/, Annotation-Training400/, '
                        'REFUGE-Validation400*/). Optional but recommended.')
    p.add_argument('--g1020_dir',  default=None,
                   help='Path to G1020 dataset root (Images/, Masks/). Optional but recommended.')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--out_dir',    default='models/od_detector')
    p.add_argument('--encoder',    required=True,
                   help='No default on purpose -- the b2-for-lesion-detectors choice was '
                        'never validated for OD (a structurally different, easier task with '
                        'more combined data). Compare e.g. efficientnet-b0 vs efficientnet-b2 '
                        'explicitly rather than assuming continuity.')
    p.add_argument('--img_size',   type=int,   default=512,
                   help='Whole-image resize target. No patch extraction -- the disc is one '
                        'large blob per image, unlike sparse small lesions.')
    p.add_argument('--epochs',     type=int,   default=60)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--alpha',      type=float, default=0.5,
                   help='Tversky alpha (FN weight). 0.5 = balanced; the disc has no severe '
                        'class-imbalance/small-blob problem the way sparse lesions do.')
    p.add_argument('--target_dice', type=float, default=0.90)
    p.add_argument('--patience',   type=int,   default=10)
    p.add_argument('--drive_dir',  default=None,
                   help='rclone remote path to back up best checkpoint on each improvement.')
    p.add_argument('--resume',     default=None)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
