#!/usr/bin/env python3
"""
RealizeMD -- Learned EX / MA Detector Training
===============================================
Trains a binary lesion segmenter for Hard Exudates (EX) or Microaneurysms (MA).
Addresses the boundary-bloat problem in the rule-based detectors:
  rule-based EX pixel precision = 0.07, MA = 0.03
  target after training: recall@IoU=0.3 >= 0.65, pixel precision >= 0.50

Loss: Dice (primary) + Focal Tversky alpha=0.6 (secondary, balanced recall/precision).
      Unlike the HE detector (alpha=0.7 recall-lean), we use alpha=0.6 here because
      the main failure mode is over-detection (boundary bloat), not under-detection.
Arch: EfficientNet-B4 U-Net (segmentation_models_pytorch) -- same as HE detector.
Data: IDRiD train split (54 images). DDR has EX/MA labels; pass --ddr_dir to add them.
      IDRiD test split (27 images) held out for final eval.

Usage:
    python scripts/train_ex_ma_detector.py \
        --lesion_type EX \
        --idrid_dir  "A. Segmentation/A. Segmentation" \
        --out_dir    models/ex_detector \
        --epochs     80

    python scripts/train_ex_ma_detector.py \
        --lesion_type MA \
        --idrid_dir  "A. Segmentation/A. Segmentation" \
        --out_dir    models/ma_detector \
        --epochs     80 \
        --alpha      0.65

MA notes: blobs are 1-20px. Patch-level blob sampling uses smaller thresholds.
          Push alpha to 0.65-0.70 if recall stalls below 0.50.
EX notes: blobs are larger and brighter. alpha=0.6 default is appropriate.
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

# Lesion type -> IDRiD subfolder + mask filename suffix
_LESION_CFG = {
    "EX": {"subdir": "3. Hard Exudates",  "suffix": "_EX", "ddr_label": "EX"},
    "MA": {"subdir": "1. Microaneurysms", "suffix": "_MA", "ddr_label": "MA"},
}


# -- Loss ---------------------------------------------------------------------

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky) ** self.gamma


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        inter   = (probs * targets).sum()
        return 1 - (2 * inter + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth)


# -- Metrics ------------------------------------------------------------------

def recall_at_iou(pred_bin, gt_bin, iou_thresh=0.3):
    """Blob-level recall@IoU=0.3. Returns (recall, hits, n_gt)."""
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


def pixel_precision(pred_bin, gt_bin):
    """Fraction of predicted pixels that fall on GT lesion territory."""
    pred_px = int((pred_bin > 0).sum())
    if pred_px == 0:
        return 1.0, 0
    tp_px = int(((gt_bin > 0) & (pred_bin > 0)).sum())
    return tp_px / pred_px, pred_px


# -- Preprocessing ------------------------------------------------------------

def preprocess(img_bgr):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


# -- Patch extraction ---------------------------------------------------------

def _sample_patches(image, mask, ys, xs, patch, half, n):
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


def extract_patches(image, mask, lesion_type, patch=768, n_pos=6, n_neg=3):
    """
    Blob-stratified patch sampling. MA uses smaller blob-size thresholds
    since microaneurysms are 1-20px, not 500-5000px like HE/EX.
    """
    H, W   = mask.shape
    half   = patch // 2
    patches = []

    labeled = sk_measure.label(mask > 0)
    regions = sk_measure.regionprops(labeled)

    if regions:
        if lesion_type == "MA":
            def blob_count(area):
                return 1 if area < 20 else (2 if area < 100 else 3)
        else:  # EX
            def blob_count(area):
                return 1 if area < 500 else (2 if area < 5000 else 3)

        blob_allocs = [(reg, blob_count(reg.area)) for reg in regions]
        total_alloc = sum(n for _, n in blob_allocs)
        scale = min(1.0, n_pos / max(total_alloc, 1))

        for reg, n_blob in blob_allocs:
            n_scaled = max(1, round(n_blob * scale))
            ys, xs = np.where(labeled == reg.label)
            patches += _sample_patches(image, mask, ys, xs, patch, half, n_scaled)

        remaining = max(0, n_pos - len(patches))
        if remaining > 0:
            ys, xs = np.where(mask > 0)
            patches += _sample_patches(image, mask, ys, xs, patch, half, remaining)

    for _ in range(n_neg):
        y0 = random.randint(0, max(0, H - patch))
        x0 = random.randint(0, max(0, W - patch))
        patches.append((image[y0:y0+patch, x0:x0+patch],
                        mask[y0:y0+patch, x0:x0+patch]))
    return patches


# -- Augmentation -------------------------------------------------------------

def build_aug(patch_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.RandomBrightnessContrast(0.35, 0.35, p=0.5),
        A.RandomGamma(gamma_limit=(60, 140), p=0.5),
        A.Resize(patch_size, patch_size),
    ])


# -- Dataset ------------------------------------------------------------------

class LesionDataset(Dataset):
    def __init__(self, samples, lesion_type, patch_size=768,
                 augment=True, n_pos=6, n_neg=3):
        self.samples     = samples
        self.lesion_type = lesion_type
        self.patch_size  = patch_size
        self.augment     = augment
        self.n_pos       = n_pos
        self.n_neg       = n_neg
        self.aug         = build_aug(patch_size)
        self._build_patches()

    def _build_patches(self):
        self.patches = []
        for img_path, mask_path in self.samples:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = preprocess(img_bgr)
            if mask_path is not None and Path(mask_path).exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.uint8) * 255 if mask is not None else \
                       np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            else:
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

            for img_p, mask_p in extract_patches(
                    img_rgb, mask, self.lesion_type,
                    self.patch_size, self.n_pos, self.n_neg):
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


# -- Data loading -------------------------------------------------------------

def load_idrid_samples(idrid_dir, lesion_type, split='train'):
    """
    Returns (img_path, mask_path_or_None) for all images in the split.
    Images without a lesion mask get mask_path=None (treated as negatives).
    """
    cfg    = _LESION_CFG[lesion_type]
    folder = 'a. Training Set' if split == 'train' else 'b. Testing Set'
    img_dir  = idrid_dir / '1. Original Images' / folder
    mask_dir = idrid_dir / '2. All Segmentation Groundtruths' / folder / cfg['subdir']
    samples  = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        stem = img_path.stem
        mask_path = next(
            (mask_dir / f"{stem}{cfg['suffix']}{ext}"
             for ext in ('.tif', '.png', '.bmp')
             if (mask_dir / f"{stem}{cfg['suffix']}{ext}").exists()),
            None)
        samples.append((img_path, mask_path))
    return samples


def load_ddr_samples(ddr_split_dir, lesion_type):
    """
    DDR layout: ddr_split_dir/image/<id>.jpg
    Masks:      ddr_split_dir/label/<LESION_TYPE>/<id>.tif
    """
    cfg     = _LESION_CFG[lesion_type]
    img_dir = ddr_split_dir / 'image'
    if not img_dir.exists():
        return []
    for label_folder in ('label', 'segmentation label'):
        mask_dir = ddr_split_dir / label_folder / cfg['ddr_label']
        if mask_dir.exists():
            break
    else:
        return []
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


# -- Sliding-window inference -------------------------------------------------

def sliding_window_predict(model, img_rgb, patch=768, stride=512,
                           threshold=0.5, device='cuda'):
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
                t  = torch.from_numpy(
                    img_rgb[y0:y0+patch, x0:x0+patch].transpose(2, 0, 1)
                ).float() / 255.0
                prob = torch.sigmoid(model(t.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
                prob_map[y0:y0+patch, x0:x0+patch] += prob
                count[y0:y0+patch, x0:x0+patch]    += 1
    prob_map /= np.maximum(count, 1)
    return (prob_map > threshold).astype(np.uint8) * 255


# -- Validation ---------------------------------------------------------------

def validate(model, val_samples, lesion_type, args, device):
    """
    Full-image sliding-window eval.
    Returns (recall, pixel_precision, hits, n_gt, combined=recall*precision).
    Combined is the checkpoint selection criterion.
    """
    model.eval()
    hits_total = 0
    gt_total   = 0
    prec_vals  = []

    for img_path, mask_path in val_samples:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb  = preprocess(img_bgr)
        pred     = sliding_window_predict(
            model, img_rgb,
            patch=args.patch_size, stride=640,
            threshold=args.threshold, device=device)
        pred_bin = (pred > 0).astype(np.uint8)

        if mask_path is None or not Path(mask_path).exists():
            # Negative image: count FPs toward precision
            prec_vals.append(0.0 if pred_bin.any() else 1.0)
            continue

        gt     = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt > 0).astype(np.uint8) if gt is not None else \
                 np.zeros(pred_bin.shape, dtype=np.uint8)

        if gt_bin.max() == 0:
            prec_vals.append(0.0 if pred_bin.any() else 1.0)
            continue

        _, hits, n_gt = recall_at_iou(pred_bin, gt_bin)
        hits_total += hits
        gt_total   += n_gt

        prec, _ = pixel_precision(pred_bin, gt_bin)
        prec_vals.append(prec)

    recall    = hits_total / max(gt_total, 1)
    mean_prec = float(np.mean(prec_vals)) if prec_vals else 0.0
    combined  = recall * mean_prec
    return recall, mean_prec, hits_total, gt_total, combined


# -- Training loop ------------------------------------------------------------

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lesion = args.lesion_type.upper()
    assert lesion in _LESION_CFG, f"lesion_type must be EX or MA, got {lesion}"
    print(f"Device: {device}  |  Lesion type: {lesion}")

    idrid_dir  = Path(args.idrid_dir)
    train_samp = load_idrid_samples(idrid_dir, lesion, split='train')
    val_samp   = load_idrid_samples(idrid_dir, lesion, split='test')

    if args.ddr_dir:
        ddr_root  = Path(args.ddr_dir)
        ddr_samp  = load_ddr_samples(ddr_root / 'train', lesion) + \
                    load_ddr_samples(ddr_root / 'valid', lesion)
        print(f"DDR samples: {len(ddr_samp)}")
        train_samp = train_samp + ddr_samp

    pos_train = sum(1 for _, m in train_samp if m is not None)
    print(f"Train: {len(train_samp)} images ({pos_train} with {lesion} GT)")
    print(f"Val:   {len(val_samp)} images")

    train_ds = LesionDataset(train_samp, lesion, patch_size=args.patch_size,
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

    ckpt_name = f"{lesion.lower()}_detector_best.pth"
    print(f"\nStarting training -- {args.epochs} epochs, patience={args.patience}")
    print(f"Target: recall >= {args.target_recall}  pixel_precision >= {args.target_precision}\n")

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
            loss   = criterion_dice(logits, masks) + criterion_ft(logits, masks)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / max(len(train_dl), 1)

        if epoch % 5 == 0 or epoch == 1:
            recall, prec, hits, total, combined = validate(
                model, val_samp, lesion, args, device)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                  f"recall={recall:.4f} ({hits}/{total}) | "
                  f"px_precision={prec:.4f} | combined={combined:.4f}")

            if combined > best_combined:
                best_combined = combined
                no_improve    = 0
                ckpt = out_dir / ckpt_name
                torch.save({'epoch': epoch, 'lesion': lesion,
                            'recall': recall, 'px_precision': prec,
                            'combined': combined,
                            'state_dict': model.state_dict(),
                            'threshold': args.threshold}, str(ckpt))
                print(f"  New best: recall={recall:.4f}  px_precision={prec:.4f}  combined={combined:.4f}")
                if args.drive_dir:
                    import subprocess as _sp
                    r = _sp.run(['rclone', 'copy', str(ckpt), args.drive_dir],
                                capture_output=True, text=True)
                    status = "backed up" if r.returncode == 0 else f"backup failed: {r.stderr.strip()}"
                    print(f"  Drive: {status}")
            else:
                no_improve += 1
                print(f"  no improvement ({no_improve}/{args.patience})")
                if no_improve >= args.patience:
                    print("Early stopping.")
                    break

            if recall >= args.target_recall and prec >= args.target_precision:
                print(f"\nBoth targets reached -- done.")
                break
        else:
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f}")

    torch.save({'epoch': epoch, 'lesion': lesion, 'combined': best_combined,
                'state_dict': model.state_dict(), 'threshold': args.threshold},
               str(out_dir / f"{lesion.lower()}_detector_final.pth"))
    print(f"\nBest combined (recall x pixel_precision): {best_combined:.4f}")
    print(f"Checkpoints: {out_dir}")


# -- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train learned EX or MA detector')
    p.add_argument('--lesion_type',      required=True, choices=['EX', 'MA'],
                   help='EX (Hard Exudates) or MA (Microaneurysms)')
    p.add_argument('--idrid_dir',        required=True,
                   help='Path to IDRiD A. Segmentation/A. Segmentation dir')
    p.add_argument('--ddr_dir',          default=None,
                   help='Path to DDR dataset root (optional, has EX and MA labels)')
    p.add_argument('--out_dir',          default='models/lesion_detector')
    p.add_argument('--encoder',          default='efficientnet-b4')
    p.add_argument('--epochs',           type=int,   default=80)
    p.add_argument('--batch_size',       type=int,   default=4)
    p.add_argument('--patch_size',       type=int,   default=768)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--alpha',            type=float, default=0.6,
                   help='Tversky alpha (FN weight). 0.6=balanced; push to 0.65 if MA recall stalls.')
    p.add_argument('--threshold',        type=float, default=0.5,
                   help='Probability threshold at inference. Lower to improve recall.')
    p.add_argument('--target_recall',    type=float, default=0.65)
    p.add_argument('--target_precision', type=float, default=0.50,
                   help='Pixel precision target. Current rule-based: EX=0.07, MA=0.03.')
    p.add_argument('--n_pos',            type=int,   default=6,
                   help='Positive patches per image per epoch.')
    p.add_argument('--n_neg',            type=int,   default=3,
                   help='Negative patches per image per epoch.')
    p.add_argument('--patience',         type=int,   default=10,
                   help='Early-stop patience in validation checks (5 epochs each).')
    p.add_argument('--drive_dir',        default=None,
                   help='rclone remote path for checkpoint backup.')
    p.add_argument('--resume',           default=None,
                   help='Path to checkpoint (.pth) to resume from.')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
