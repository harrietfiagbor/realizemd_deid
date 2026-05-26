#!/bin/bash
# scripts/setup_runpod.sh
# One-shot environment setup for RunPod GPU instance
# Run ONCE after spinning up the instance:
#   bash scripts/setup_runpod.sh

set -e
echo "=== RealizeMD De-identification Pipeline — RunPod Setup ==="

# ── System deps ───────────────────────────────────────────────────────────────
apt-get update -q
apt-get install -q -y git libgl1-mesa-glx libglib2.0-0 p7zip-full unzip

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
    timm \
    kaggle

echo "✅ Python packages installed"

# ── Download EyePACS test images ──────────────────────────────────────────────
export KAGGLE_USERNAME="adjoadede33"
export KAGGLE_KEY="KGAT_8abb244b54efbba9afc7f3a802af4408"

mkdir -p /workspace/data/eyepacs/images
kaggle competitions download -c diabetic-retinopathy-detection \
    -f test.zip.001 -p /workspace/data/

# This handles the nested zip wrapper you found
unzip -o /workspace/data/test.zip.001.zip -d /workspace/data/eyepacs/

# This extracts the JPEGs flat into /workspace/data/eyepacs/images/
7z e /workspace/data/eyepacs/test.zip.001 -o/workspace/data/eyepacs/images/ "*.jpeg" -r -y || true

# Cleanup to keep your workspace from getting full
rm -f /workspace/data/test.zip.001.zip /workspace/data/eyepacs/test.zip.001

echo "Done: $(ls /workspace/data/eyepacs/images/ | wc -l) images"

# ── Download Model A weights (arkanivasarkar Attention U-Net) ─────────────────
mkdir -p /workspace/models
ln -sfn /workspace/models /workspace/realizemd_deid/models
git clone --quiet \
    https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET \
    /workspace/arkan_unet
cp -r "/workspace/arkan_unet/Trained models/" /workspace/models/attention_unet/
echo "✅ Model A weights ready at /workspace/models/attention_unet/"
echo "   Available: $(ls '/workspace/models/attention_unet/')"

# ── Download RETFound ─────────────────────────────────────────────────────────
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
# lama-cleaner auto-downloads on first use — pre-warm here to avoid cold start
python -c "
from lama_cleaner.model import LaMa
m = LaMa(device='cuda')
print('LaMa weights ready')
"
echo "✅ LaMa weights ready"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete — everything is ready ==="
echo ""
echo "De-identify:"
echo "  python scripts/run_pipeline.py \\"
echo "      --input   /workspace/data/eyepacs/images/ \\"
echo "      --output  /workspace/data/deid/ \\"
echo "      --weights '/workspace/models/attention_unet/AttentionUNet.h5' \\"
echo "      --device  cuda"