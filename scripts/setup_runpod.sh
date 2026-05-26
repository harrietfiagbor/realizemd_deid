#!/bin/bash
# scripts/setup_runpod.sh
# One-shot environment setup for RunPod GPU instance
# Run once after spinning up the instance

set -e
echo "=== RealizeMD De-identification Pipeline — RunPod Setup ==="

# ── System deps ───────────────────────────────────────────────────────────────
apt-get update -q
apt-get install -q -y git libgl1-mesa-glx libglib2.0-0 p7zip-full

# ── Python deps ───────────────────────────────────────────────────────────────
pip install -q \
    torch torchvision \
    tensorflow \
    opencv-python-headless \
    scikit-image \
    scikit-learn \
    matplotlib \
    Pillow \
    tqdm \
    pandas \
    numpy \
    omegaconf \
    huggingface_hub \
    transformers \
    lpips \
    pytorch-fid \
    lama-cleaner \
    pyyaml \
    einops \
    timm

echo "✅ Python packages installed"

# ── Download Model A weights (arkanivasarkar Attention U-Net) ─────────────────
mkdir -p /workspace/models
git clone --quiet https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET \
    /workspace/arkan_unet
cp -r "/workspace/arkan_unet/Trained models/" /workspace/models/attention_unet/
echo "✅ Model A weights ready at /workspace/models/attention_unet/"

# ── Download RETFound weights ─────────────────────────────────────────────────
git clone --quiet https://github.com/rmaphoh/RETFound /workspace/RETFound
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='YukunZhou/RETFound_mae_natureCFP',
    filename='RETFound_mae_natureCFP.pth',
    local_dir='/workspace/models/'
)
print('RETFound weights:', path)
"
echo "✅ RETFound weights ready"

# ── Download LaMa weights ─────────────────────────────────────────────────────
# lama-cleaner downloads weights automatically on first run
# Pre-download to avoid cold-start delay
python -c "
from lama_cleaner.model import LaMa
m = LaMa(device='cuda')
print('LaMa weights ready')
"
echo "✅ LaMa weights ready"

echo ""
echo "=== Setup complete ==="
echo "Run: python scripts/run_pipeline.py --input /data/images/ --output /data/deid/"
