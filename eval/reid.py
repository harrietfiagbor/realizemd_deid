"""
eval/reid.py
Privacy metric — Re-identification rate using RETFound embeddings.
Implements Section 2.1 of the supervisor's eval framework.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


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
    Compute Rank-1 re-identification rate.

    For each de-identified image, find the closest original in embedding space.
    Rank-1 re-id = fraction where closest match is the correct patient.

    Args:
        original_embeddings: {stem: (1024,) embedding}
        deid_embeddings:     {stem: (1024,) embedding}

    Returns dict:
        rank1_rate:      fraction correctly re-identified
        random_baseline: 1/N
        ratio_vs_random: rank1_rate / random_baseline
        pass:            bool (≤ 2× random baseline)
        per_image:       list of per-image result dicts
        same_patient_auc: AUC for same vs different patient pair classification
    """
    stems = list(deid_embeddings.keys())
    orig_keys = list(original_embeddings.keys())
    orig_matrix = np.stack([original_embeddings[k] for k in orig_keys])  # (N, 1024)

    n = len(stems)
    random_baseline = 1 / len(orig_keys)
    per_image = []
    n_correct = 0

    for stem in stems:
        if stem not in deid_embeddings:
            continue
        e_deid = deid_embeddings[stem]

        # Cosine similarity to all originals
        norms = np.linalg.norm(orig_matrix, axis=1) * np.linalg.norm(e_deid) + 1e-8
        sims = (orig_matrix @ e_deid) / norms

        top1_key = orig_keys[np.argmax(sims)]
        correct = (top1_key == stem)
        if correct:
            n_correct += 1

        self_sim = float(sims[orig_keys.index(stem)]) if stem in orig_keys else None

        per_image.append({
            'stem':          stem,
            'top1_match':    top1_key,
            'top1_correct':  correct,
            'self_similarity': self_sim,
            'top1_sim':      float(np.max(sims)),
            'mean_sim':      float(np.mean(sims)),
        })

    rank1_rate = n_correct / n if n > 0 else 0
    target = 2 * random_baseline

    # Same-patient AUC
    same_patient_auc = _compute_same_patient_auc(
        original_embeddings, deid_embeddings
    )

    return {
        'n':                 n,
        'rank1_rate':        round(rank1_rate, 4),
        'random_baseline':   round(random_baseline, 4),
        'ratio_vs_random':   round(rank1_rate / random_baseline, 2),
        'pass':              rank1_rate <= target,
        'same_patient_auc':  round(same_patient_auc, 4),
        'per_image':         per_image,
    }


def _compute_same_patient_auc(original_embeddings: dict,
                               deid_embeddings: dict) -> float:
    """
    AUC for same-patient vs different-patient pair cosine similarity.
    Target: ≤ 0.55 (near random).
    """
    from sklearn.metrics import roc_auc_score

    stems = [s for s in deid_embeddings if s in original_embeddings]
    labels, scores = [], []

    for i, s1 in enumerate(stems):
        e1 = deid_embeddings[s1]
        for j, s2 in enumerate(stems):
            e2 = original_embeddings[s2]
            sim = float(
                np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            )
            labels.append(1 if s1 == s2 else 0)
            scores.append(sim)

    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)
