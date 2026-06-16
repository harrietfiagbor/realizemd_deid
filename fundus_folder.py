"""
src/dataset/fundus_folder.py

Simple image-folder dataset that plugs into Mueller's training pipeline
without requiring EyePACS metadata CSV.

For Phase 0: healthy fundus images only, no subspace labels needed.
For Phase 1: add disease labels via the optional label_csv argument.

Usage in config:
    dataset_class: fundus_folder   # instead of eyepacs
    data:
      image_root_dir: /workspace/fundus_train/
      image_size: 128
      batch_size: 6
      num_workers: 4
      label_csv:          # optional, leave empty for Phase 0
      label_columns: []   # e.g. ["dr_grade"] for Phase 1
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}


class FundusFolderDataset(Dataset):
    """
    Minimal image-folder dataset for Mueller's StyleGAN2 pipeline.

    Returns dicts with:
        'image'  : float tensor (C, H, W) in [-1, 1]  — StyleGAN convention
        'labels' : list of int tensors, one per subspace (empty list if no labels)
        'idx'    : int, sample index
    """

    def __init__(
        self,
        image_root_dir: str,
        image_size: int = 128,
        label_csv: Optional[str] = None,
        label_columns: Optional[List[str]] = None,
        split: str = 'train',
        train_val_split: float = 0.9,
        seed: int = 42,
        subset: Optional[int] = None,
        input_preprocessing: bool = True,
    ):
        self.image_size = image_size
        self.label_columns = label_columns or []

        # Collect image paths
        root = Path(image_root_dir)
        paths = sorted([
            p for p in root.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        ])
        if not paths:
            raise FileNotFoundError(f'No images found in {image_root_dir}')

        # Train/val split
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(paths))
        n_train = int(len(idx) * train_val_split)
        if split == 'train':
            idx = idx[:n_train]
        else:
            idx = idx[n_train:]

        self.paths = [paths[i] for i in idx]

        # Optional subset
        if subset is not None and subset < len(self.paths):
            self.paths = self.paths[:subset]

        # Optional label CSV
        self.labels_df = None
        self._num_classes = {}
        if label_csv and label_columns:
            df = pd.read_csv(label_csv, index_col=0)
            self.labels_df = df
            for col in label_columns:
                self._num_classes[col] = int(df[col].nunique())
        else:
            # Phase 0: no labels — subspace dims will be [] 
            for col in self.label_columns:
                self._num_classes[col] = 1

        # Transforms: resize + centre crop + normalize to [-1, 1]
        if input_preprocessing:
            self.transform = T.Compose([
                T.Resize(int(image_size * 1.1)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        print(f'  FundusFolderDataset [{split}]: {len(self.paths)} images, '
              f'size={image_size}, labels={self.label_columns}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img_t = self.transform(img)

        labels = []
        if self.labels_df is not None and self.label_columns:
            stem = path.stem
            if stem in self.labels_df.index:
                row = self.labels_df.loc[stem]
                for col in self.label_columns:
                    labels.append(torch.tensor(int(row[col]), dtype=torch.long))
            else:
                for col in self.label_columns:
                    labels.append(torch.tensor(0, dtype=torch.long))

        return {
            'image':  img_t,
            'labels': labels,
            'idx':    idx,
        }


def get_fundus_dataloaders(cfg: dict, split_seed: int = 42):
    """
    Build train and val dataloaders from config dict.
    Drop-in replacement for Mueller's EyePACS dataloader builder.
    """
    data_cfg = cfg.get('data', {})

    common = dict(
        image_root_dir=data_cfg['image_root_dir'],
        image_size=data_cfg.get('image_size', 128),
        label_csv=data_cfg.get('label_csv', None),
        label_columns=data_cfg.get('label_columns', []),
        seed=split_seed,
        input_preprocessing=data_cfg.get('input_preprocessing', True),
    )

    train_ds = FundusFolderDataset(
        split='train',
        subset=data_cfg.get('train_subset', None),
        **common,
    )
    val_ds = FundusFolderDataset(
        split='val',
        subset=data_cfg.get('val_subset', None),
        **common,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=data_cfg.get('batch_size', 6),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=data_cfg.get('batch_size', 6),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=False,
    )

    return train_dl, val_dl, train_ds
