"""
eval/reid.py
Privacy metric — Re-identification rate using RETFound embeddings.
Implements Section 2.1 of the supervisor's eval framework.

Patient-aware: EyePACS images are paired (left/right per patient).
A re-id hit = matching to *any* image of the same patient, not just
the exact same laterality.
"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


def _patient_id(stem: str) -> str:
    """Extract patient ID from an EyePACS-style filename stem.

    Handles patterns like:
      '12345_left'  -> '12345'
      '12345_right' -> '12345'
      '12345_left_deid' -> '12345'
      '12345'       -> '12345'  (no laterality suffix)
    """
    # Strip known pipeline suffixes first
    s = re.sub(r'(_deid(-checkpoint)?)', '', stem)
    # Strip laterality
    s = re.sub(r'[_-](left|right)$', '', s, flags=re.IGNORECASE)
    return s


_retfound = None
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_retfound(weights_path: str, retfound_dir: str, device: str = 'cuda'):
    """Load RETFound ViT-Large as a pure feature extractor (no classification head)."""
    global _retfound
    import sys
    sys.path.insert(0, retfound_dir)
    import models_vit

    # Find the right constructor — API varies by repo version
    constructor = None
    for name in ['RETFound_mae', 'RETFound_dinov2', 'vit_large_patch16', 'vit_large', 'create_model']:
        if hasattr(models_vit, name):
            constructor = getattr(models_vit, name)
            break

    if constructor is None:
        raise RuntimeError(
            f'Cannot find ViT constructor in models_vit. '
            f'Available: {[x for x in dir(models_vit) if not x.startswith("_")]}'
        )

    model = constructor(num_classes=0, global_pool=True)

    # Monkeypatch forward_features to be compatible with newer timm versions
    try:
        import inspect
        original_forward_features = model.forward_features
        sig = inspect.signature(original_forward_features)
        if 'attn_mask' not in sig.parameters and 'kwargs' not in sig.parameters:
            def patched_forward_features(x, *args, **kwargs):
                return original_forward_features(x)
            model.forward_features = patched_forward_features
            print("✅ Patched RETFound forward_features for timm compatibility")
    except Exception as patch_err:
        print(f"Note: Could not patch forward_features: {patch_err}")

    checkpoint = torch.load(weights_path, map_location='cpu')

    # Handle different checkpoint formats
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    _retfound = {'model': model, 'device': device}
    print(f'✅ RETFound loaded as embedder (device={device})')
    return _retfound


@torch.no_grad()
def embed(img_rgb: np.ndarray) -> np.ndarray:
    """Get 1024-dim RETFound embedding from an RGB image."""
    if _retfound is None:
        raise RuntimeError('RETFound not loaded. Call reid.load_retfound() first.')
    device = _retfound['device']
    tensor = _transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    if hasattr(_retfound['model'], 'forward_features'):
        emb = _retfound['model'].forward_features(tensor)
    else:
        emb = _retfound['model'](tensor)
    
    emb = emb.squeeze()
    if emb.ndim > 1:
        emb = emb[0]  # Take the CLS token if sequence is still present
        
    return emb.cpu().numpy()  # (1024,)


def embed_batch(image_dict: dict) -> dict:
    """
    Embed a dict of {stem: img_rgb} images.
    Returns {stem: embedding (1024,)}
    """
    embeddings = {}
    for stem, img in tqdm(image_dict.items(), desc='Embedding'):
        embeddings[stem] = embed(img)
    return embeddings


def compute_reid_rate(original_embeddings: dict,
                      deid_embeddings: dict) -> dict:
    """
    Compute patient-aware Rank-1 re-identification rate.

    For each de-identified image, find the closest original in embedding space.
    A match counts as correct if the top-1 hit belongs to the **same patient**
    (either eye), not just the exact same image stem.

    Args:
        original_embeddings: {stem: (1024,) embedding}
        deid_embeddings:     {stem: (1024,) embedding}

    Returns dict:
        n_images:        number of de-identified images evaluated
        n_patients:      number of unique patients
        rank1_rate:      fraction correctly re-identified (patient-level)
        random_baseline: 1 / n_patients
        ratio_vs_random: rank1_rate / random_baseline
        pass:            bool (≤ 2× random baseline)
        same_patient_auc: AUC for same vs different patient pair classification
        per_image:       list of per-image result dicts
    """
    stems = list(deid_embeddings.keys())
    orig_keys = list(original_embeddings.keys())
    orig_matrix = np.stack([original_embeddings[k] for k in orig_keys])  # (N, 1024)

    # Build patient-ID lookup for originals
    orig_pid = {k: _patient_id(k) for k in orig_keys}

    n = len(stems)
    unique_patients = set(orig_pid.values())
    n_patients = len(unique_patients)
    random_baseline = 1 / n_patients if n_patients > 0 else 0
    per_image = []
    n_correct = 0

    for stem in stems:
        e_deid = deid_embeddings[stem]
        pid = _patient_id(stem)

        # Cosine similarity to all originals
        norms = np.linalg.norm(orig_matrix, axis=1) * np.linalg.norm(e_deid) + 1e-8
        sims = (orig_matrix @ e_deid) / norms

        top1_idx = int(np.argmax(sims))
        top1_key = orig_keys[top1_idx]
        top1_pid = orig_pid[top1_key]
        correct = (top1_pid == pid)
        if correct:
            n_correct += 1

        # Self-similarity: best sim to any original of the SAME patient
        same_patient_idxs = [i for i, k in enumerate(orig_keys) if orig_pid[k] == pid]
        self_sim = float(max(sims[i] for i in same_patient_idxs)) if same_patient_idxs else None

        per_image.append({
            'stem':            stem,
            'patient_id':      pid,
            'top1_match':      top1_key,
            'top1_patient':    top1_pid,
            'top1_correct':    correct,
            'self_similarity':  self_sim,
            'top1_sim':        float(np.max(sims)),
            'mean_sim':        float(np.mean(sims)),
        })

    rank1_rate = n_correct / n if n > 0 else 0
    target = 2 * random_baseline

    # Same-patient AUC
    same_patient_auc = _compute_same_patient_auc(
        original_embeddings, deid_embeddings
    )

    return {
        'n_images':          n,
        'n_patients':        n_patients,
        'rank1_rate':        round(rank1_rate, 4),
        'random_baseline':   round(random_baseline, 4),
        'ratio_vs_random':   round(rank1_rate / random_baseline, 2) if random_baseline > 0 else 0,
        'pass':              rank1_rate <= target,
        'same_patient_auc':  round(same_patient_auc, 4),
        'per_image':         per_image,
    }


def _compute_same_patient_auc(original_embeddings: dict,
                               deid_embeddings: dict) -> float:
    """
    AUC for same-patient vs different-patient pair cosine similarity.
    Patient-aware: left/right images of the same patient are positive pairs.
    Target: ≤ 0.55 (near random).
    """
    from sklearn.metrics import roc_auc_score

    deid_stems = list(deid_embeddings.keys())
    orig_stems = list(original_embeddings.keys())
    labels, scores = [], []

    for s1 in deid_stems:
        pid1 = _patient_id(s1)
        e1 = deid_embeddings[s1]
        for s2 in orig_stems:
            pid2 = _patient_id(s2)
            e2 = original_embeddings[s2]
            sim = float(
                np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            )
            labels.append(1 if pid1 == pid2 else 0)
            scores.append(sim)

    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)
