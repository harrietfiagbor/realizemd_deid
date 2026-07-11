#!/usr/bin/env python3
"""
RealizeMD -- Learned EX Detector Training (exploration-grade)
==============================================================
Trains a binary hard-exudate segmenter to test whether a learned EX detector
produces coherent enough blobs to survive 256px downsampling, the way the
learned MA detector did. The rule-based EX detector (Fix 1+2, threshold=45)
is sub-pixel at 256px; this checks whether that is a rule-based-detector
problem or a fundamental EX problem, mirroring the MA exploration.

Exploration-grade, single pass, same approach as train_ma_detector.py.
Not intended to be matured -- run once, put through the 256px/512px
resampling eval, then park per the same plan as MA.

Loss: Focal Tversky (alpha=0.6, balanced) + Dice. Lower alpha than MA (0.8)
      because EX's rule-based failure mode was over-detection (boundary
      bloat), not under-detection -- no need to push hard on recall here.
Arch: EfficientNet-B2 U-Net (segmentation_models_pytorch).
Data: IDRiD-train + DDR lesion_segmentation (EX only). IDRiD-test held out.

Usage (RunPod or local GPU):
    python scripts/train_ex_detector.py \
        --idrid_dir  "/workspace/data/idrid/A. Segmentation" \
        --ddr_dir    /workspace/data/ddr/DDR-dataset/lesion_segmentation \
        --out_dir    /workspace/models/ex_detector \
        --epochs     80 \
        --patch_size 512 \
        --batch_size 8
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
from skimage import measure as sk_measure

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """alpha=0.6: balanced, since EX's failure mode is over-detection, not
    under-detection (unlike MA which needed alpha=0.8)."""
    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
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


# ── Metric ───────────────────────────────────────────────────────────────────

def pixel_preservation(pred_bin: np.ndarray, gt_bin: np.ndarray):
    total = int(gt_bin.sum())
    if total == 0:
        return None
    return float((gt_bin & pred_bin).sum()) / total


def recall_at_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, iou_thresh: float = 0.3):
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
            inter    = (gt_blob & (pred_crop == pl)).sum()
            union    = gt_reg.area + pred_areas.get(pl, 0) - inter
            best_iou = max(best_iou, inter / max(union, 1))
        if best_iou >= iou_thresh:
            hits += 1
    return hits / len(gt_regions), hits, len(gt_regions)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_for_training(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


# ── Patch extraction ──────────────────────────────────────────────────────────

def _sample_patches_from_coords(image, mask, ys, xs, patch, half, n):
    patches = []
    H, W = mask.shape
    for _ in range(n):
        i  = random.randrange(len(ys))
        cy = int(np.clip(ys[i] + random.randint(-half // 2, half // 2), half, H - half))
        cx = int(np.clip(xs[i] + random.randint(-half // 2, half // 2), half, W - half))
        y0, x0 = cy - half, cx - half
        patches.append((image[y0:y0+patch, x0:x0+patch],
                        mask[y0:y0+patch, x0:x0+patch]))
    return patches


def extract_patches(image: np.ndarray, mask: np.ndarray,
                    patch: int = 512, n_pos: int = 8, n_neg: int = 4):
    """EX blobs are 500-5000px+, larger than MA -- fewer patches needed per
    blob since each patch covers proportionally more of the lesion."""
    H, W    = mask.shape
    half    = patch // 2
    patches = []

    labeled = sk_measure.label(mask > 0)
    regions = sk_measure.regionprops(labeled)

    if regions:
        def blob_count(area):
            return 1 if area < 500 else (2 if area < 5000 else 3)

        blob_allocs = [(reg, blob_count(reg.area)) for reg in regions]
        total_alloc = sum(n for _, n in blob_allocs)
        scale = min(1.0, n_pos / max(total_alloc, 1))

        for reg, n_blob in blob_allocs:
            n_scaled = max(1, round(n_blob * scale))
            blob_ys, blob_xs = np.where(labeled == reg.label)
            patches += _sample_patches_from_coords(
                image, mask, blob_ys, blob_xs, patch, half, n_scaled)

        remaining = max(0, n_pos - len(patches))
        if remaining > 0:
            ex_ys, ex_xs = np.where(mask > 0)
            patches += _sample_patches_from_coords(
                image, mask, ex_ys, ex_xs, patch, half, remaining)

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
        A.RandomBrightnessContrast(0.35, 0.35, p=0.5),
        A.RandomGamma(gamma_limit=(60, 140), p=0.5),
        A.Resize(patch_size, patch_size),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class EXDataset(Dataset):
    def __init__(self, samples: list, patch_size: int = 512,
                 augment: bool = True, n_pos: int = 8, n_neg: int = 4):
        self.samples    = samples
        self.patch_size = patch_size
        self.augment    = augment
        self.n_pos      = n_pos
        self.n_neg      = n_neg
        self.aug        = build_aug(patch_size)
        self._build_patches()

    def _build_patches(self):
        self.patches = []
        for item in self.samples:
            img_path, mask_path = item[0], item[1]
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
    folder  = 'a. Training Set' if split == 'train' else 'b. Testing Set'
    img_dir = idrid_dir / '1. Original Images' / folder
    ex_dir  = idrid_dir / '2. All Segmentation Groundtruths' / folder / '3. Hard Exudates'
    samples = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        stem    = img_path.stem
        ex_glob = list(ex_dir.glob(f'{stem}_EX.*'))
        if not ex_glob:
            continue
        samples.append((img_path, ex_glob[0]))
    return samples


def load_ddr_samples(ddr_split_dir: Path):
    """DDR split layout: ddr_split_dir/image/<id>.jpg
    Masks: ddr_split_dir/label/EX/<id>.tif (train)
        or ddr_split_dir/segmentation label/EX/<id>.tif (valid)."""
    img_dir = ddr_split_dir / 'image'
    if not img_dir.exists():
        return []
    for label_folder in ('label', 'segmentation label'):
        mask_dir = ddr_split_dir / label_folder / 'EX'
        if mask_dir.exists():
            break
    samples = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        for ext in ('.tif', '.png'):
            mask_path = mask_dir / (img_path.stem + ext)
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() > 0:
                    samples.append((img_path, mask_path))
                break
    return samples


# ── Sliding-window inference ──────────────────────────────────────────────────

def sliding_window_predict(model: nn.Module, img_rgb: np.ndarray,
                           patch: int = 512, stride: int = 384,
                           threshold: float = 0.35, device: str = 'cuda') -> np.ndarray:
    model.eval()
    H, W     = img_rgb.shape[:2]
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
                t = torch.from_numpy(patch_img.transpose(2, 0, 1)).float() / 255.0
                t = t.unsqueeze(0).to(device)
                prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                prob_map[y0:y0+patch, x0:x0+patch] += prob
                count[y0:y0+patch, x0:x0+patch]    += 1
    prob_map /= np.maximum(count, 1)
    return (prob_map > threshold).astype(np.uint8) * 255


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: nn.Module, val_samples: list, args, device: str):
    model.eval()
    hits_total = 0
    gt_total   = 0
    pres_vals  = []
    for item in val_samples:
        img_path, mask_path = item[0], item[1]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb  = preprocess_for_training(img_bgr)
        pred     = sliding_window_predict(
            model, img_rgb, patch=args.patch_size, stride=args.patch_size - 128,
            threshold=args.threshold, device=device)
        pred_bin = (pred > 0).astype(np.uint8)
        gt       = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_bin   = (gt > 0).astype(np.uint8)
        _, hits, n_gt = recall_at_iou(pred_bin, gt_bin)
        hits_total += hits
        gt_total   += n_gt
        p = pixel_preservation(pred_bin, gt_bin)
        if p is not None:
            pres_vals.append(p)
    recall    = hits_total / max(gt_total, 1)
    mean_pres = float(np.mean(pres_vals)) if pres_vals else 0.0
    return recall, hits_total, gt_total, mean_pres


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    idrid_dir  = Path(args.idrid_dir)
    train_samp = load_idrid_samples(idrid_dir, split='train')
    val_samp   = load_idrid_samples(idrid_dir, split='test')

    if args.ddr_dir:
        ddr_root  = Path(args.ddr_dir)
        ddr_train = load_ddr_samples(ddr_root / 'train')
        ddr_valid = load_ddr_samples(ddr_root / 'valid')
        ddr_samp  = ddr_train + ddr_valid
        print(f"DDR samples: {len(ddr_samp)} ({len(ddr_train)} train + {len(ddr_valid)} valid)")
        train_samp = train_samp + ddr_samp

    print(f"Train samples (with EX GT): {len(train_samp)}")
    print(f"Val   samples (with EX GT): {len(val_samp)}")

    train_ds = EXDataset(train_samp, patch_size=args.patch_size,
                        augment=True, n_pos=args.n_pos, n_neg=args.n_neg)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)

    model = smp.Unet(
        encoder_name    = args.encoder,
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,
        decoder_dropout = 0.2,
    ).to(device)
    print(f"Model: Unet + {args.encoder}")

    criterion_ft   = FocalTverskyLoss(alpha=args.alpha, beta=1 - args.alpha)
    criterion_dice = DiceLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_combined = 0.0
    no_improve    = 0
    start_epoch   = 1

    if args.resume:
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r['state_dict'])
        best_combined = ckpt_r.get('combined', 0.0)
        start_epoch   = ckpt_r.get('epoch', 0) + 1
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1}  best_combined={best_combined:.4f}")

    print(f"\nStarting training — {args.epochs} epochs, patience={args.patience}")
    print(f"Target: recall@IoU=0.3 >= {args.target_recall}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.resample()
        train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)

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
            recall, hits, total, mean_pres = validate(model, val_samp, args, device)
            combined = recall * mean_pres
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                  f"recall={recall:.4f} ({hits}/{total}) | "
                  f"preservation={mean_pres:.4f} | combined={combined:.4f}")

            if combined > best_combined:
                best_combined = combined
                no_improve    = 0
                ckpt = out_dir / 'ex_detector_best.pth'
                torch.save({'epoch': epoch, 'recall': recall,
                            'preservation': mean_pres, 'combined': combined,
                            'state_dict': model.state_dict(),
                            'threshold': args.threshold}, str(ckpt))
                print(f"  New best saved: recall={recall:.4f}  preservation={mean_pres:.4f}  combined={combined:.4f}")
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
                print(f"  no improvement ({no_improve}/{args.patience})")
                if no_improve >= args.patience:
                    print("Early stopping.")
                    break

            if recall >= args.target_recall:
                print(f"\nTarget recall {args.target_recall} reached — done.")
                break
        else:
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

    torch.save({'epoch': epoch, 'combined': best_combined,
                'state_dict': model.state_dict(),
                'threshold': args.threshold},
               str(out_dir / 'ex_detector_final.pth'))
    print(f"\nBest combined (recall x preservation): {best_combined:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train learned EX detector (exploration-grade)')
    p.add_argument('--idrid_dir',     required=True)
    p.add_argument('--ddr_dir',       default=None)
    p.add_argument('--out_dir',       default='models/ex_detector')
    p.add_argument('--encoder',       default='efficientnet-b2')
    p.add_argument('--epochs',        type=int,   default=80)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--patch_size',    type=int,   default=512)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--alpha',         type=float, default=0.6,
                   help='Tversky alpha (FN weight). Lower than MA (0.8) -- '
                        'EX failure mode is over-detection, not under-detection.')
    p.add_argument('--threshold',     type=float, default=0.35)
    p.add_argument('--target_recall', type=float, default=0.95,
                   help='Set high to avoid premature stop (learned from MA run1).')
    p.add_argument('--n_pos',         type=int,   default=6)
    p.add_argument('--n_neg',         type=int,   default=3)
    p.add_argument('--patience',      type=int,   default=10)
    p.add_argument('--drive_dir',     default=None)
    p.add_argument('--resume',        default=None)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
